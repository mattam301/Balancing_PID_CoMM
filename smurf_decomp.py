import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityBranch(nn.Module):
    """One modality branch: maps input to 3 outputs (private, shared1, shared2)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_private = nn.Linear(in_dim, out_dim)
        self.fc_shared1 = nn.Linear(in_dim, out_dim)
        self.fc_shared2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_hat   = self.fc_private(x)
        y_hat_1 = self.fc_shared1(x)
        y_hat_2 = self.fc_shared2(x)
        return y_hat, y_hat_1, y_hat_2


class ThreeModalityModel(nn.Module):
    def __init__(self, in_dim, out_dim, final_dim):
        super().__init__()
        # Three modality branches
        self.mod1 = ModalityBranch(in_dim, out_dim)
        self.mod2 = ModalityBranch(in_dim, out_dim)
        self.mod3 = ModalityBranch(in_dim, out_dim)

        # Final fusion layer
        self.fusion = nn.Linear(out_dim * 10, final_dim)  
        # 9 outputs from modalities + 1 extra branch = 10

    def forward(self, x1, x2, x3, x_other):
        # Get modality-specific outputs
        m1 = self.mod1(x1)  # (y_hat, y_hat_1, y_hat_2)
        m2 = self.mod2(x2)
        m3 = self.mod3(x3)

        # Flatten all outputs
        all_mod_outs = torch.cat([*m1, *m2, *m3], dim=-1)  # 9 vectors concat



        # Final fusion
        final_input = torch.cat([all_mod_outs, x_other], dim=-1)
        final_repr = self.fusion(final_input)

        return m1, m2, m3, final_repr


def compute_corr_loss(m1, m2, m3):
    """
    m1, m2, m3: each is a tuple of (hat, hat_1, hat_2)
       - hat   : independent head
       - hat_1 : shared with the *next* modality
       - hat_2 : shared with the *other* modality
    Each element is a [batch, dim] tensor.
    """

    # unpack
    m1_hat, m1_hat1, m1_hat2 = m1
    m2_hat, m2_hat1, m2_hat2 = m2
    m3_hat, m3_hat1, m3_hat2 = m3

    # ========== Uncorrelation loss ==========
    # within-modality: independent vs shared
    unco_pairs = [
        (m1_hat, m1_hat1), (m1_hat, m1_hat2),
        (m2_hat, m2_hat1), (m2_hat, m2_hat2),
        (m3_hat, m3_hat1), (m3_hat, m3_hat2)
    ]
    L_unco = sum(torch.mean(torch.abs(torch.cov(torch.cat([a.T, b.T], dim=0))[0, 1])) for a, b in unco_pairs) / len(unco_pairs)

    # ========== Cross-modal correlation loss ==========
    # matching shared heads across modalities
    cross_pairs = [
        (m1_hat1, m2_hat1),
        (m1_hat2, m3_hat1),
        (m2_hat2, m3_hat2)
    ]
    cor_terms = []
    for a, b in cross_pairs:
        # covariance matrix
        cov = torch.cov(torch.cat([a.T, b.T], dim=0))
        cov_ab = cov[0, 1]
        var_a = cov[0, 0]
        var_b = cov[1, 1]

        cor_terms.append((-1.0) * cov_ab + 0.5 * var_a * var_b)

    L_cor = sum(cor_terms) / len(cor_terms)

    # final correlation loss
    corr_loss = L_unco + L_cor
    return corr_loss, L_unco, L_cor