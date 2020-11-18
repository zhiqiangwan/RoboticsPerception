import numpy as np

from utils import est_homography


def warp_pts(video_ref_pts, logo_pts, sample_pts):
    """Warp the points in sample_pts to points in the logo image.

    Use the correspondence between video_ref_pts and logo_pts to estimate
    the homography. Then, warp the sample_pts with the estimated homography.

    Args:
        video_ref_pts: array with shape of (4, 2). (x, y) coordinates of
        reference points in the video image.
        logo_pts: array with shape of (4, 2). (x, y) coordinates of points
        in the logo image.
        sample_pts: array with shape of (N, 2).

    Returns:
        warped_pts: array with shape of (N, 2). (x, y) coordinates of warpped
        sample points.

    """
    H = est_homography(video_ref_pts, logo_pts)
    num_sample_pts = np.shape(sample_pts)[0]
    sample_pts_hm = np.concatenate((sample_pts, np.ones((num_sample_pts, 1))), axis=1)
    sample_pts_hm = np.transpose(sample_pts_hm)
    warped_pts_hm = np.matmul(H, sample_pts_hm)
    warped_pts = np.divide(warped_pts_hm[:2, :], warped_pts_hm[2, :])
    warped_pts = np.transpose(warped_pts)
    return warped_pts


def update_RGB(video_img, logo_img, video_pts, logo_pts):
    """Update the RGB value of the video_pts in video_img.
    """
    height, width, _ = np.shape(logo_img)
    logo_pts_max = np.ones_like(logo_pts)
    logo_pts_max[:, 0] = height
    logo_pts_max[:, 1] = width
    logo_pts = np.rint(logo_pts)
    logo_pts = np.clip(logo_pts, 0, logo_pts_max - 1).astype(np.int)

    video_img[video_pts[:, 0], video_pts[:, 1], :] = logo_img[logo_pts[:, 0], logo_pts[:, 1], :]
    return video_img
