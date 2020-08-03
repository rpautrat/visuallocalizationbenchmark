import torch
import torch.nn.functional as func
from adalam import AdalamFilter


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


# Symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def ratio_matcher(descriptors1, descriptors2, ratio=0.8):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # Symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()


# Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def mutual_nn_ratio_matcher(descriptors1, descriptors2, ratio=0.8):
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()


# Mutual nearest neighbors matcher for lisrd
def lisrd_matcher(descriptors1, descriptors2, meta_descriptors1,
                   meta_descriptors2):
    device = descriptors1.device
    desc_weights = torch.einsum('nid,mid->nim', (meta_descriptors1, meta_descriptors2))
    del meta_descriptors1, meta_descriptors2
    desc_weights = func.softmax(desc_weights, dim=1)
    desc_sims = torch.einsum('nid,mid->nim',
                             (descriptors1, descriptors2)) * desc_weights
    del descriptors1, descriptors2, desc_weights
    desc_sims = torch.sum(desc_sims, dim=1)
    nn12 = torch.max(desc_sims, dim=1)[1]
    nn21 = torch.max(desc_sims, dim=0)[1]
    ids1 = torch.arange(desc_sims.shape[0], dtype=torch.long, device=device)
    del desc_sims
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()


# Mutual nearest neighbors matcher for lisrd
def sequential_lisrd_matcher(descriptors1, descriptors2, meta_descriptors1,
                             meta_descriptors2):
    device = descriptors1.device
    desc_sims = 0.
    weights_sum = 0.
    for i in range(4):
        weights = torch.exp(meta_descriptors1[:, i, :]
                            @ meta_descriptors2[:, i, :].t())
        weights_sum += weights
        desc_sims += (descriptors1[:, i, :] @ descriptors2[:, i, :].t()) * weights
    del meta_descriptors1, meta_descriptors2, descriptors1, descriptors2, weights
    desc_sims /= weights_sum
    del weights_sum
    nn12 = torch.max(desc_sims, dim=1)[1]
    nn21 = torch.max(desc_sims, dim=0)[1]
    ids1 = torch.arange(0, desc_sims.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    return matches.data.cpu().numpy()


# Nearest neighbors matcher for lisrd, followed by Adalam filtering
def adalam_matcher(kp1, kp2, desc1, desc2, meta_desc1,
                   meta_desc2, img_shape1, img_shape2):
    device = desc1.device
    desc_weights = torch.einsum('nid,mid->nim', (meta_desc1, meta_desc2))
    del meta_desc1, meta_desc2
    desc_weights = func.softmax(desc_weights, dim=1)
    desc_sims = torch.einsum('nid,mid->nim', (desc1, desc2)) * desc_weights
    del desc1, desc2, desc_weights
    desc_sims = 2 - 2 * torch.sum(desc_sims, dim=1)

    # Compute putative matches and mutual neighbors
    dd12, nn12 = torch.topk(desc_sims, k=2, dim=1, largest=False)
    putative_matches = nn12[:, 0]
    scores = dd12[:, 0] / dd12[:, 1].clamp_min_(1e-3)
    dd21, nn21 = torch.min(desc_sims, dim=0)
    mnn = nn21[putative_matches] == torch.arange(kp1.shape[0], device=device)
    del desc_sims

    # Filter the matches with Adalam
    matcher = AdalamFilter()
    matches = matcher.filter_matches(
        kp1, kp2, putative_matches, scores, mnn, img_shape1, img_shape2,
        None, None, None, None)
    return matches.data.cpu().numpy()


# Nearest neighbors matcher for lisrd, followed by Adalam filtering
def sequential_adalam_matcher(kp1, kp2, desc1, desc2, meta_desc1,
                              meta_desc2, img_shape1, img_shape2):
    device = desc1.device
    desc_sims = 0.
    weights_sum = 0.
    for i in range(4):
        weights = torch.exp(meta_desc1[:, i, :] @ meta_desc2[:, i, :].t())
        weights_sum += weights
        desc_sims += (desc1[:, i, :] @ desc2[:, i, :].t()) * weights
    del meta_desc1, meta_desc2, desc1, desc2, weights
    desc_sims /= weights_sum
    del weights_sum
    desc_sims = 2 - 2 * torch.sum(desc_sims, dim=1)

    # Compute putative matches and mutual neighbors
    dd12, nn12 = torch.topk(desc_sims, k=2, dim=1, largest=False)
    putative_matches = nn12[:, 0]
    scores = dd12[:, 0] / dd12[:, 1].clamp_min_(1e-3)
    dd21, nn21 = torch.min(desc_sims, dim=0)
    mnn = nn21[putative_matches] == torch.arange(kp1.shape[0], device=device)
    del desc_sims

    # Filter the matches with Adalam
    matcher = AdalamFilter()
    matches = matcher.filter_matches(
        kp1, kp2, putative_matches, scores, mnn, img_shape1, img_shape2,
        None, None, None, None)
    return matches.data.cpu().numpy()