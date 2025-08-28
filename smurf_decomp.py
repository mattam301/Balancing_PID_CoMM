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
        x = x.to(next(self.parameters()).device)
        y_hat   = self.fc_private(x)
        y_hat_1 = self.fc_shared1(x)
        y_hat_2 = self.fc_shared2(x)
        return y_hat, y_hat_1, y_hat_2


class ThreeModalityModel(nn.Module):
    def __init__(self, in_dim, out_dim, final_dim, use_comm):
        super().__init__()
        # Three modality branches
        self.mod1 = ModalityBranch(in_dim, out_dim)
        self.mod2 = ModalityBranch(in_dim, out_dim)
        self.mod3 = ModalityBranch(in_dim, out_dim)
        if use_comm:
            self.fusion = nn.Linear(out_dim*3, final_dim)
        else: 
            self.fusion = nn.Linear(out_dim*2, final_dim)

    def forward(self, x1, x2, x3, x_other):
        # Get modality-specific outputs
        m1 = self.mod1(x1)  # (y_hat, y_hat_1, y_hat_2)
        m2 = self.mod2(x2)
        m3 = self.mod3(x3)

        # summing all output
        all_mod_outs = m1[0] + m1[1] + m1[2] + m2[0] + m2[1] + m2[2] + m3[0] + m3[1] + m3[2]

        # Final fusion
        all_mod_outs = all_mod_outs.to(next(self.parameters()).device)
        x_other = x_other.to(next(self.parameters()).device)
        final_input = torch.cat([all_mod_outs, x_other], dim=-1)
        final_repr = self.fusion(final_input)

        return m1, m2, m3, final_repr


def safe_cov(a, b):
    # flatten batch and seq if needed
    a = a.reshape(-1, a.shape[-1])  # [total_samples, dim]
    b = b.reshape(-1, b.shape[-1])

    # transpose to [dim, total_samples]
    a = a.T
    b = b.T

    # concat along variables
    x = torch.cat([a, b], dim=0)  # [2*dim, total_samples]
    return torch.cov(x)

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
    unco_pairs = [
        (m1_hat, m1_hat1), (m1_hat, m1_hat2),
        (m2_hat, m2_hat1), (m2_hat, m2_hat2),
        (m3_hat, m3_hat1), (m3_hat, m3_hat2)
    ]

    L_unco = sum(
        torch.mean(torch.abs(safe_cov(a, b)[0, 1]))
        for a, b in unco_pairs
    ) / len(unco_pairs)

    # ========== Cross-modal correlation loss ==========
    cross_pairs = [
        (m1_hat1, m2_hat1),
        (m1_hat2, m3_hat1),
        (m2_hat2, m3_hat2)
    ]

    cor_terms = []
    for a, b in cross_pairs:
        cov = safe_cov(a, b)
        cov_ab = cov[0, 1]
        var_a = cov[0, 0]
        var_b = cov[1, 1]
        cor_terms.append((-1.0) * cov_ab + 0.5 * var_a * var_b)

    L_cor = sum(cor_terms) / len(cor_terms)

    # final correlation loss
    corr_loss = L_unco + L_cor
    return corr_loss, L_unco, L_cor