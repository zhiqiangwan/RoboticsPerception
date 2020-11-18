from matplotlib.path import Path
import numpy as np


def calculate_interior_pts(image_size, video_ref_pts):
    """Calculate the points inside a shape defined by video_ref_pts.

    Args:
        image_size: size of video image.
        video_ref_pts: array with shape of (4, 2).

    Reeturns:
        interior_pts: array with shape of (M, 2). M is the number of points
        inside the shape.

    """

    x, y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    x, y = x.flatten(), y.flatten()
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    points = np.concatenate((x, y), axis=1)
    polygon = Path(video_ref_pts)
    mask = polygon.contains_points(points)
    interior_pts = points[mask, :]
    return interior_pts


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
