import numpy as np
import cv2


def EstimateFundamentalMatrix(x1, x2):
    """Estimate fundamental matrix. x2'*F*x1 = 0

    Args:
        x1: shape (N, 2) array. Keypoints in image1.
        x2: shape (N, 2) array. Corresponding Keypoints in image2.

    Returns:
        F: shape (3, 3) array with rank 2.
    """
    N = x1.shape[0]
    A = np.ones((N, 9))
    A[:, 0] = x2[:, 0] * x1[:, 0]
    A[:, 1] = x2[:, 0] * x1[:, 1]
    A[:, 2] = x2[:, 0]
    A[:, 3] = x2[:, 1] * x1[:, 0]
    A[:, 4] = x2[:, 1] * x1[:, 1]
    A[:, 5] = x2[:, 1]
    A[:, 6] = x1[:, 0]
    A[:, 7] = x1[:, 1]

    _, _, V_T = np.linalg.svd(A)
    H = V_T[-1, :]
    H = np.reshape(H, (3, 3))

    # Constrain the rank of H to 2
    U, S, V_T = np.linalg.svd(H)
    S[-1] = 0
    S = np.diag(S)
    F = np.matmul(np.matmul(U, S), V_T)
    F = F / np.linalg.norm(F, 2)
    return F


def EssentialMatrixFromFundamentalMatrix(F, K):
    """Estimate essential matrix with fundamental matrix F
    and instrinsic parameter K. E = K^T * F * K
    """
    E = np.matmul(np.matmul(np.transpose(K), F), K)
    U, _, V_T = np.linalg.svd(E)
    # The first two sigular values of E are equal. The third one is 0.
    S = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    E = np.matmul(np.matmul(U, S), V_T)
    return E


def LinearTriangulation(K, R2, C2, x1, x2):
    """Estimate the location of 3D points. The cross product of x and P*X is 0.

    Args:
        K: shape (3, 3) array. Instrinsic parameter.
        R2: shape (3, 3) array. Rotation matrix of the second camera.
        C2: shape (3, 1) vector. position of the second camera.
        x1: shape (N, 2) array. Keypoints in image1. 
        x2: shape (N, 2) array. Corresponding keypoints in image2.
    Returns:
        X_3d: Shape (N, 3) array. Coordinates of 3D points.
    """
    # Camera 1: P = [K | 0]
    P = np.zeros((3, 4))
    P[:, :3] = K
    P1 = np.expand_dims(P[0, :], axis=0) # shape (1, 4)
    P2 = np.expand_dims(P[1, :], axis=0)
    P3 = np.expand_dims(P[2, :], axis=0)
    # Camera 2: P_   =  K  * R2 * [I | -C2]
    #          (3,4)  (3,3) (3,3)  (3,4)
    I_C = np.zeros((3, 4))
    I_C[:, :3] = np.eye(3)
    I_C[:, 3] = -np.squeeze(C2)
    P_ = np.matmul(np.matmul(K, R2), I_C)
    P1_ = np.expand_dims(P_[0, :], axis=0) # shape (1, 4)
    P2_ = np.expand_dims(P_[1, :], axis=0)
    P3_ = np.expand_dims(P_[2, :], axis=0)  

    N = x1.shape[0]
    x = np.expand_dims(x1[:, 0], axis=1) # shape (N, 1)
    y = np.expand_dims(x1[:, 1], axis=1)
    x_ = np.expand_dims(x2[:, 0], axis=1)
    y_ = np.expand_dims(x2[:, 1], axis=1)
    x_mul_P3 = np.matmul(x, P3) # shape (N, 4)
    y_mul_P3 = np.matmul(y, P3)
    x_prime_mul_P3 = np.matmul(x_, P3_)
    y_prime_mul_P3 = np.matmul(y_, P3_)

    X_3d = np.zeros((N, 3))
    for i in range(N):
        A = np.zeros((4, 4))
        A[0, :] = x_mul_P3[i, :] - P1
        A[1, :] = y_mul_P3[i, :] - P2
        A[2, :] = x_prime_mul_P3[i, :] - P1_
        A[3, :] = y_prime_mul_P3[i, :] - P2_

        _, _, V_T = np.linalg.svd(A)
        X_hm = V_T[-1, :]
        X_3d[i, :] = X_hm[:3] / X_hm[3]

    # Use opencv to do triangulation. This is used to test this function.
    # X_3d_cv2 = cv2.triangulatePoints(P, P_, np.transpose(x1), np.transpose(x2))
    # X_3d_cv2[:3, :] = X_3d_cv2[:3, :] / np.tile(X_3d_cv2[3, :], (3, 1))
    return X_3d


def RegisterImage(X, x3, K):
    """Register new image into the reconstructed coordinate system.
    Given the location of 3D points X and the corresponding points
    in the image x3. Estimate the projection matrix P such that x3 = P*X.
    Then, extract camera pose from P.

    Args:
        X: shape (N, 3). Coordinates of 3D points.
        x3: shape (N, 2). Corresponding keypoints in the new image.
        K: instrinsic parameter.

    Returns:
        R3: shape (3, 3). Rotation matrix of the camera.
        C3: shape (3, 1). Position of the camera.
    """
    N = X.shape[0]
    A = np.zeros((2*N, 12))
    X_hm = np.concatenate((X, np.ones((N, 1))), axis=1)
    x_mul_X = np.expand_dims(x3[:, 0], axis=1) * X_hm
    y_mul_X = np.expand_dims(x3[:, 1], axis=1) * X_hm
    A[:N, 4:8] = -X_hm
    A[:N, 8:] = y_mul_X
    A[N:, :4] = X_hm
    A[N:, 8:] = -x_mul_X

    _, _, V_T = np.linalg.svd(A)
    P = V_T[-1, :]
    P = P / P[-1]
    P = np.reshape(P, (3, 4))

    P_a = np.matmul(np.linalg.inv(K), P)
    R_b = P_a[:, :3]
    U, S, V_T = np.linalg.svd(R_b)

    det_UV = np.linalg.det(U) * np.linalg.det(V_T)
    if det_UV > 0:
        R3 = np.matmul(U, V_T)
        t = P_a[:, 3] / S[0]
        C3 = -np.matmul(np.transpose(R3), t)
    else:
        R3 = -np.matmul(U, V_T)
        t = -P_a[:, 3] / S[0]
        C3 = -np.matmul(np.transpose(R3), t)  

    # Use opencv to test this function.
    # _, R3_cv, t3_cv = cv2.solvePnP(X, x3, K, None)
    return R3, np.expand_dims(C3, axis=1)


def ReprojectToImage(X, K, R2, C2, R3, C3):
    """Reproject 3D points to image1, image2, image3.

    Args:
        X: shape (N, 3) array. Coordinates of 3D points.
        K: shape (3, 3) array. Instrinsic parameter.
        R2: shape (3, 3) array. Rotation matrix of camera 2.
        C2: shape (3, 1) vector. Position of camera 2.
        R3: shape (3, 3) array. Rotation matrix of camera 3.
        C3: shape (3, 1) vector. Position of camera 3.        
    
    Returns:
        x1p: shape (N, 2) array. Coordinates of projected points in image 1.
        x2p: shape (N, 2) array. Coordinates of projected points in image 2.
        x3p: shape (N, 2) array. Coordinates of projected points in image 3.
    """
    N = np.shape(X)[0]
    x1p_hm = np.transpose(np.matmul(K, np.transpose(X)))
    x1p = x1p_hm[:, :2] / np.expand_dims(x1p_hm[:, 2], axis=1)  # (N, 2) / (N, 1) where the denominator will be broadcasded to (N, 2)
    X_minus_C = np.transpose(X) - C2
    x2p_hm = np.transpose(np.matmul(np.matmul(K, R2), X_minus_C))
    x2p = x2p_hm[:, :2] / np.expand_dims(x2p_hm[:, 2], axis=1)
    X_minus_C = np.transpose(X) - C3
    x3p_hm = np.transpose(np.matmul(np.matmul(K, R3), X_minus_C))
    x3p = x3p_hm[:, :2] / np.expand_dims(x3p_hm[:, 2], axis=1)

    # Use opencv to test this function. 
    # rvec1 = cv2.Rodrigues(np.eye(3))[0]
    # tvec1 = np.zeros(3)
    # rvec2 = cv2.Rodrigues(R2)[0]
    # tvec2 = -np.matmul(R2, C2)
    # rvec3 = cv2.Rodrigues(R3)[0]
    # tvec3 = -np.matmul(R3, C3)
    # x1p_cv = cv2.projectPoints(X, rvec1, tvec1, K, None)[0]
    # x2p_cv = cv2.projectPoints(X, rvec2, tvec2, K, None)[0]
    # x3p_cv = cv2.projectPoints(X, rvec3, tvec3, K, None)[0]
    return x1p, x2p, x3p


def DisplayProjectedPoints(img, x, x_p):
    """Dispaly the original and projected points in the image.
    """
    N = x.shape[0]
    x = np.rint(x).astype(np.int)
    x_p = np.rint(x_p).astype(np.int)
    img_ = img.copy()
    for i in range(N):
        orginal_point = (x[i, 0], x[i, 1])
        projected_point = (x_p[i, 0], x_p[i, 1])
        img_ = cv2.line(img_, orginal_point, projected_point, (255, 0, 0), 1)
    return img_


def Jacobian_Triangulation(K, R, C, X):
    """Jacobian matrix for nonlinear triangulation.

    Args:
        K: shape (3, 3) array. Instrinsic parameter.
        R: shape (3, 3) array. Rotation matrix.
        C: shape (3, 1) vector. Position of camera.
        X: shape (3,) vector. Coordinates of 3D points before refinement.

    Returns:
        J: shape (2, 3) array.
    """
    x = np.matmul(np.matmul(K, R), np.expand_dims(X, axis=1) - C)
    u = x[0]
    v = x[1]
    w = x[2]
    f = K[0, 0]
    p_x = K[0, 2]
    p_y = K[1, 2]

    dev_u_X = np.array([f*R[0, 0] + p_x * R[2, 0],
                        f*R[0, 1] + p_x*R[2, 1],
                        f*R[0, 2] + p_x*R[2, 2]])
    dev_v_X = np.array([f*R[1, 0] + p_y*R[2, 0],
                        f*R[1, 1] + p_y*R[2, 1],
                        f*R[1, 2] + p_y*R[2, 2]])
    dev_w_X = R[2, :]
    J1 = (w * dev_u_X - u * dev_w_X) / (w * w)
    J2 = (w * dev_v_X - v * dev_w_X) / (w * w)
    J = np.zeros((2, 3))
    J[0] = J1
    J[1] = J2
    return J


def Single_Point_Nonlinear_Triangulation(K, R2, C2, R3, C3, x1, x2, x3, X):
    """Given three camera poses and linearly triangulated points, X, refine the locations of
    a single 3D point that minimizes reprojection error.

    Args:
        K: shape (3, 3) array. Instrinsic parameter.
        R2: shape (3, 3) array. Rotation matrix of camera 2.
        C2: shape (3, 1) vector. Position of camera 2.
        R3: shape (3, 3) array. Rotation matrix of camera 3.
        C3: shape (3, 1) vector. Position of camera 3.
        x1: shape (2,) vector. Coordinates of one keypoint in image 1.       
        x2: shape (2,) vector. Coordinates of one keypoint in image 2. 
        x3: shape (2,) vector. Coordinates of one keypoint in image 3. 
        X: shape (3,) vector. Coordinates of the 3D point before refinement.
    
    Returns:
        X_refine: shape (3,) vector. Coordinates of the 3D point after refinement.
    """    
    R1 = np.eye(3)
    C1 = np.zeros((3, 1))
    J1 = Jacobian_Triangulation(K, R1, C1, X)  # shape (2, 3)
    J2 = Jacobian_Triangulation(K, R2, C2, X)
    J3 = Jacobian_Triangulation(K, R2, C2, X)
    J = np.concatenate((J1, J2, J3), axis=0)

    x1p = np.matmul(K, X)
    x2p = np.matmul(np.matmul(K, R2), X - np.squeeze(C2))
    x3p = np.matmul(np.matmul(K, R3), X - np.squeeze(C3))
    f = np.array([
        x1p[0] / x1p[2],
        x1p[1] / x1p[2],
        x2p[0] / x2p[2],
        x2p[1] / x2p[2],
        x3p[0] / x3p[2],
        x3p[1] / x3p[2]
    ])
    b = np.array([
        x1[0],
        x1[1],
        x2[0],
        x2[1],
        x3[0],
        x3[1]
    ])

    J_T_J_inv_J_T = np.matmul(np.linalg.inv(np.matmul(np.transpose(J), J)), np.transpose(J))
    X_refine = X + np.matmul(J_T_J_inv_J_T, b - f)
    return X_refine


def Nonlinear_Triangulation(K, R2, C2, R3, C3, x1, x2, x3, X):
    """Given three camera poses and linearly triangulated points, X, refine the locations of
    the 3D points that minimizes reprojection error.

    Args:
        K: shape (3, 3) array. Instrinsic parameter.
        R2: shape (3, 3) array. Rotation matrix of camera 2.
        C2: shape (3, 1) vector. Position of camera 2.
        R3: shape (3, 3) array. Rotation matrix of camera 3.
        C3: shape (3, 1) vector. Position of camera 3.
        x1: shape (N, 2) array. Coordinates of keypoints in image 1.       
        x2: shape (N, 2) array. Coordinates of keypoints in image 2. 
        x3: shape (N, 2) array. Coordinates of keypoints in image 3. 
        X: shape (N, 3) array. Coordinates of 3D points before refinement.
    
    Returns:
        X_refine: shape (N, 3) array. Coordinates of 3D points after refinement.
    """
    N = np.shape(X)[0]
    X_refine = np.zeros((N, 3))

    for i in range(N):
        X0 = X[i, :]
        for iteration in range(3):
            X0 = Single_Point_Nonlinear_Triangulation(K, R2, C2, R3, C3, x1[i], x2[i], x3[i], X0)
        X_refine[i] = X0
    return X
