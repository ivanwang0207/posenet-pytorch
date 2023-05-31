
import numpy as np

_EPS = np.finfo(float).eps * 4.0

def euler_to_quaternion(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Converts Euler angles representing spatial rotations to equivalent representations by unit quaterians. This
    function is vectorized and operates on arrays of angles.

    Args:
        roll (np.ndarray): Nx1 array of roll angle in degrees for N observations pitch (np.ndarray): Nx1 array of pitch
        angle in degrees for N observations yaw (np.ndarray): Nx1 array of yaw angle in degrees for N observations

    Returns:
        np.ndarray: Nx4 array, with N rows of observations and the four columns being x, y, z, w quaterion components.
    """
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([x, y, z, w]).transpose()


def transformation_matrix(t: np.ndarray, q: np.ndarray):
    """Generate 4x4 homogeneous transformation matrices from arrays of 3D point translations and unit quaternion
    representations of rotations. This function is vectorized and operates on arrays of transformation parameters.

    Args:
        t (np.ndarray): Nx3 array of translations (tx, ty, tz)
        q (np.ndarray): Nx4 array of unit quaternions (qx, qy, qz, qw)

    Returns:
        np.ndarray: Nx4x4 array of 4x4 transformation matrices
    """
    # Allocate transformation matrices with only translation
    arr_transforms = np.repeat(np.eye(4)[None, :, :], t.shape[0], axis=0)
    arr_transforms[:, :3, 3] = t

    nq = np.square(np.linalg.norm(q, axis=1))
    mask = nq >= _EPS  # mask for rotations of magnitude greater than epsilon

    # For transformations with non-zero rotation, calculate rotation matrix
    q = np.sqrt(2.0 / nq)[:, None] * q
    q = q[:, :, None] * q[:, None, :]  # outer product
    arr_transforms[mask, 0, 0] = 1.0 - q[mask, 1, 1] - q[mask, 2, 2]
    arr_transforms[mask, 0, 1] = q[mask, 0, 1] - q[mask, 2, 3]
    arr_transforms[mask, 0, 2] = q[mask, 0, 2] + q[mask, 1, 3]
    arr_transforms[mask, 1, 0] = q[mask, 0, 1] + q[mask, 2, 3]
    arr_transforms[mask, 1, 1] = 1.0 - q[mask, 0, 0] - q[mask, 2, 2]
    arr_transforms[mask, 1, 2] = q[mask, 1, 2] - q[mask, 0, 3]
    arr_transforms[mask, 2, 0] = q[mask, 0, 2] - q[mask, 1, 3]
    arr_transforms[mask, 2, 1] = q[mask, 1, 2] + q[mask, 0, 3]
    arr_transforms[mask, 2, 2] = 1.0 - q[mask, 0, 0] - q[mask, 1, 1]
    return arr_transforms


def ominus(arr_a: np.ndarray, arr_b: np.ndarray) -> np.ndarray:
    """Computes the result of applying the inverse motion composition operator on two pose matrices. This implementation
    is vectorized: arr_a and arr_b should be numpy.ndarray with shape (N, 4, 4) containing N 4x4 transformation
    matrices. The output will also be N 4x4 matrices.

    Args:
        arr_a (np.ndarray): Nx4x4 array of N poses (homogeneous 4x4 matrix)
        arr_b (np.ndarray): Nx4x4 array of N poses (homogeneous 4x4 matrix)

    Returns:
        np.ndarray: Nx4x4 array of N resulting 4x4 relative transformation matrices between a and b
    """
    return np.matmul(np.linalg.inv(arr_a), arr_b)


def compute_distance(arr_transforms: np.ndarray) -> np.ndarray:
    """
    Compute the distance of the translational components of an array of N 4x4 homogeneous matrices.

    Args:
        arr_transforms (np.ndarray): Nx4x4 array of N 4x4 transformation matrices

    Returns:
        np.ndarray: 1D array of length N with distance values
    """
    return np.linalg.norm(arr_transforms[:, :3, 3], axis=1)


def compute_angle(arr_transforms: np.ndarray) -> np.ndarray:
    """
    Compute the rotation angle from an array of N 4x4 homogeneous matrices.

    Args:
        arr_transforms (np.ndarray): Nx4x4 array of N 4x4 transformation matrices

    Returns:
        np.ndarray: 1D array of length N with angle values
    """
    # an invitation to 3-d vision, p 27
    return np.arccos(
        np.minimum(
            1,
            np.maximum(
                -1,
                (np.trace(arr_transforms[:, :3, :3], axis1=1, axis2=2) - 1) / 2,
            ),
        )
    )