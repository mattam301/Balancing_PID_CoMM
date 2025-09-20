import torch
import torch.nn as nn


def batch_to_all_tva(feature_t, feature_v, feature_a, lengths, no_cuda):

    node_feature_t, node_feature_v, node_feature_a = [], [], []
    batch_size = feature_t.size(1)

    for j in range(batch_size):
        node_feature_t.append(feature_t[:lengths[j], j, :])
        node_feature_v.append(feature_v[:lengths[j], j, :])
        node_feature_a.append(feature_a[:lengths[j], j, :])

    node_feature_t = torch.cat(node_feature_t, dim=0)
    node_feature_v = torch.cat(node_feature_v, dim=0)
    node_feature_a = torch.cat(node_feature_a, dim=0)

    if not no_cuda:
        node_feature_t = node_feature_t.cuda()
        node_feature_v = node_feature_v.cuda()
        node_feature_a = node_feature_a.cuda()

    return node_feature_t, node_feature_v, node_feature_a

def info_nce_loss(rep_m: torch.Tensor, rep_a: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Compute InfoNCE loss for two batches of representations.

    Args:
        rep_m: torch.Tensor of shape [N, D]
        rep_a: torch.Tensor of shape [N, D]
        temperature: scaling factor for logits

    Returns:
        loss: scalar tensor
    """
    # Normalize to unit sphere (cosine similarity)
    rep_m = F.normalize(rep_m, dim=1)
    rep_a = F.normalize(rep_a, dim=1)

    N = rep_m.size(0)

    # Similarity matrix [N, N]
    logits = torch.matmul(rep_m, rep_a.T) / temperature

    # Targets are diagonal indices (positive pairs)
    labels = torch.arange(N, device=rep_m.device)

    # Cross entropy loss
    loss_m2a = F.cross_entropy(logits, labels)
    loss_a2m = F.cross_entropy(logits.T, labels)

    # Symmetrized loss
    loss = (loss_m2a + loss_a2m) / 2
    return loss

def augment_text(tensor, noise_level=0.1):
    # Add small Gaussian noise (simulating slight perturbation in embeddings)
    if tensor is None:
        return None
    return tensor + noise_level * torch.randn_like(tensor)

def augment_audio(tensor, noise_level=0.05):
    # Add small Gaussian noise to audio features
    if tensor is None:
        return None
    return tensor + noise_level * torch.randn_like(tensor)

def augment_visual(tensor, noise_level=0.05):
    # Add small Gaussian noise to visual features
    if tensor is None:
        return None
    return tensor + noise_level * torch.randn_like(tensor)
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i]**
                               2) * loss + torch.log(1 + self.params[i]**2)
        return loss_sum
