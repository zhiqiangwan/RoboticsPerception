import numpy as np


def est_homography(src_pts, tgt_pts):
    """Estimate the homography to transform from src_pts to tgt_pts.

    Args:
        src_pts: array with shape of (N, 2). (x, y) coordinates of source points.
        tgt_pts: array with shape of (N, 2). (x, y) coordinates of target points.

    Returns:
        H: array with shape of (3, 3).

    """
    x_src = np.expand_dims(src_pts[:, 0], axis=1)
    y_src = np.expand_dims(src_pts[:, 1], axis=1)
    x_tgt = np.expand_dims(tgt_pts[:, 0], axis=1)
    y_tgt = np.expand_dims(tgt_pts[:, 1], axis=1)
    zero_elem = np.zeros_like(x_src)
    neg_one_elem = -np.ones_like(x_src)

    A1 = np.concatenate((-x_src, -y_src, neg_one_elem, zero_elem,
                         zero_elem, zero_elem, x_src*x_tgt, y_src*x_tgt, x_tgt), axis=1)
    A2 = np.concatenate((zero_elem, zero_elem, zero_elem, -x_src,
                         -y_src, neg_one_elem, x_src*y_tgt, y_src*y_tgt, y_tgt), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    _, _, V_T = np.linalg.svd(A)
    h = V_T[-1:]
    H = np.reshape(h, (3, 3))
    return H


def est_Rt(H):
    """Given homorgraphy H, estimate rotation R and translation t. 

    Args:
        H: Homorgraphy (r1, r2, t)

    Returns:
        R: Rotation matrix (3 X 3)
        t: translation vector (3 X 1)

    """
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = np.cross(h1, h2)
    R_ = np.concatenate(
        (np.expand_dims(h1, axis=1), np.expand_dims(h2, axis=1), np.expand_dims(h3, axis=1)),
        axis=1
    )
    U, _, V_T = np.linalg.svd(R_)
    S = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, np.linalg.det(np.matmul(U, V_T))]
    ])
    R = np.matmul(np.matmul(U, S), V_T)
    t = H[:, 2] / np.linalg.norm(h1)
    return R, t
