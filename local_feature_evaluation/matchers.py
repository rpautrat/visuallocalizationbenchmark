import torch
import torch.nn.functional as func


# Mutual nearest neighbors matcher for L2 normalized descriptors.
def mutual_nn_matcher(descriptors1, descriptors2):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()


# Mutual nearest neighbors matcher for fusion_desc
def fusion_matcher(descriptors1, descriptors2, meta_descriptors1,
                   meta_descriptors2):
    device = descriptors1.device
    desc_weights = torch.sum(meta_descriptors1 * meta_descriptors2, dim=2)
    desc_weights = func.softmax(desc_weights, dim=1)
    desc_sims = torch.sum(descriptors1 * descriptors2, dim=2)
    desc_sims = torch.sum(desc_sims * desc_weights, dim=1)
    nn12 = torch.max(desc_sims, dim=1)[1]
    nn21 = torch.max(desc_sims, dim=0)[1]
    ids1 = torch.arange(0, desc_sims.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()