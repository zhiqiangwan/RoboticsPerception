"""Project a logo onto a given area in a target image.

Written for the University of Pennsylvania's Robotics:Perception course.

"""


# standard library imports
import os

# related third party imports
import cv2
from scipy import io
import numpy as np

# local application/library specific imports
from utils import calculate_interior_pts
from proect_points import warp_pts, update_RGB


logo_img = cv2.imread("images/logos/penn_engineering_logo.png")
logo_num_row = np.shape(logo_img)[0]
logo_num_col = np.shape(logo_img)[1]
# Coordinates (row_idx, col_idx) of the four corner points in logo image.
# The fiirst point is on top-left. The remaining points is organized in
# clock-wise order.
logo_pts = np.array([
    [0, 0],
    [0, logo_num_col - 1],
    [logo_num_row - 1, logo_num_col - 1],
    [logo_num_row - 1, 0]
])

# shape of video_ref_pts is (4, 2, 129). 129 test images,
# 4 points define the region to project the logo. 
# The fiirst point is on top-left. The remaining points is 
# organized in clock-wise order.
# Attention: the coordinate is (col_idx, row_idx). We need to swap these two column first.
video_ref_pts = io.loadmat("data/BarcaReal_pts.mat")['video_pts']
video_ref_pts[:, [1, 0], :] = video_ref_pts[:, [0, 1], :]
num_img = np.shape(video_ref_pts)[2]

proj_dir = "/mnt/c/research/Coding_Practice/Coursera_Robotics_Perception/RoboticsPerceptionWeek2AssignmentCode/warped_images"

for i in range(num_img):
    video_img = cv2.imread("images/barcaReal/BarcaReal{:03d}.jpg".format(i + 1))
    # Find all points in the video frame inside the polygon defined by
    # video_ref_pts
    interior_pts = calculate_interior_pts(np.shape(video_img)[:2],
                                          video_ref_pts[:, :, i])
    # Warp the interior_pts to coordinates in the logo image
    warped_logo_pts = warp_pts(video_ref_pts[:, :, i], logo_pts, interior_pts)
    # Copy the RGB values from the logo_img to the video frame
    projected_img = update_RGB(video_img, logo_img, interior_pts, warped_logo_pts)

    cv2.imwrite(os.path.join(proj_dir, "projected_img{:03d}.png".format(i + 1)), projected_img)

