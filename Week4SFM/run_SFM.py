# standard library imports
import os

# related third party imports
import scipy.io as io
import cv2
import numpy as np

# local application/library specific imports
import utils


# Load three images
data = io.loadmat("data.mat")['data'][0, 0]
img1 = data['img1']
img2 = data['img2']
img3 = data['img3']

# Load SIFT keypoints for three images
x1 = data['x1']
x2 = data['x2']
x3 = data['x3']
N = x1.shape[0]

# Load camera intrinstic parameter
K = data['K']

# Load the pose of the second camera
C2 = data['C']
R2 = data['R']

# Estimate fundamental matrix for the first two images
F = utils.EstimateFundamentalMatrix(x1, x2)

# Estimate essential matrix
E = utils.EssentialMatrixFromFundamentalMatrix(F, K)
# Recover the pose of the second camera from essential matrix
cam2_R1, cam2_R2, cam2_t = cv2.decomposeEssentialMat(E)
_, cam2_R, came2_t_, _ = cv2.recoverPose(E, x1, x2, K)

# Obtain the location of 3D points with linear triangulation
X = utils.LinearTriangulation(K, R2, C2, x1, x2)

# Register new image, i.e., estimate the pose of the third camera
R3, C3 = utils.RegisterImage(X, x3, K)

# Reproject 3D points to the images
x1p, x2p, x3p = utils.ReprojectToImage(X, K, R2, C2, R3, C3)

# Display the reprojected result
img1_ = utils.DisplayProjectedPoints(img1, x1, x1p)
img2_ = utils.DisplayProjectedPoints(img2, x2, x2p)
img3_ = utils.DisplayProjectedPoints(img3, x3, x3p)

cv2.imwrite("result/before_refine_img1.jpg", img1_)
cv2.imwrite("result/before_refine_img2.jpg", img2_)
cv2.imwrite("result/before_refine_img3.jpg", img3_)

# Refine the location of 3D points with nonlinear triangulation
X = utils.Nonlinear_Triangulation(K, R2, C2, R3, C3, x1, x2, x3, X)

# Reproject 3D points to the images
x1p, x2p, x3p = utils.ReprojectToImage(X, K, R2, C2, R3, C3)

# Display the reprojected result
img1_ = utils.DisplayProjectedPoints(img1, x1, x1p)
img2_ = utils.DisplayProjectedPoints(img2, x2, x2p)
img3_ = utils.DisplayProjectedPoints(img3, x3, x3p)

cv2.imwrite("result/after_refine_img1.jpg", img1_)
cv2.imwrite("result/after_refine_img2.jpg", img2_)
cv2.imwrite("result/after_refine_img3.jpg", img3_)

print("End!")
