# See https://github.com/NVlabs/BundleSDF/blob/master/Utils.py#L92

import numpy as np
from scipy.spatial import cKDTree

def compute_cd(GT_vertices, candidate_vertices):
    # Convert lists to numpy arrays for efficient computation
    # For each point in GT consider the closest point in PR
    # And viceversa. Sum the averages.
    kdtree1 = cKDTree(GT_vertices)
    dists1, indices1 = kdtree1.query(candidate_vertices)
    kdtree2 = cKDTree(candidate_vertices)
    dists2, indices2 = kdtree2.query(GT_vertices)
    return 0.5 * (
        dists1.mean() + dists2.mean()
    )  #!NOTE should not be mean of all, see https://pdal.io/en/stable/apps/chamfer.html

def add_err(pred, gt):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    The direct distance is considered
    """
    #   pred_pts = (pred@to_homo(model_pts).T).T[:,:3]
    #   gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
    e = np.linalg.norm(pred - gt, axis=1).mean()
    return e

def adi_err(pred_pts, gt_pts):
    """
    @pred: 4x4 mat
    @gt:
    @model: (N,3)
    For each GT point, the distnace to the closest PR point is considered
    """
    # = (pred@to_homo(model_pts).T).T[:,:3]
    # gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
    nn_index = cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
    e = nn_dists.mean()
    return e

def compute_auc(rec, max_val=0.1):
    if len(rec) == 0:
        return 0
    rec = np.sort(np.array(rec))
    n = len(rec)
    ##print(n)
    prec = np.arange(1, n + 1) / float(n)
    rec = rec.reshape(-1)
    prec = prec.reshape(-1)
    index = np.where(rec < max_val)[0]
    rec = rec[index]
    prec = prec[index]

    if len(prec) == 0:
        return 0

    mrec = [0, *list(rec), max_val]
    # Only add prec[-1] if prec is not empty
    mpre = [0, *list(prec)] + ([prec[-1]] if len(prec) > 0 else [])

    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])
    mpre = np.array(mpre)
    mrec = np.array(mrec)
    i = np.where(mrec[1:] != mrec[: len(mrec) - 1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) / max_val
    return ap