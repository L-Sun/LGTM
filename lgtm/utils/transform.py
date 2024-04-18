import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

right_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
forward_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def translate(values: np.ndarray):
    """create transform matrix from a batch translation vector 

    Args:
        values (... x 3): the batch of translation values 

    Returns:
        np.ndarray(... x 4 x 4): the transform matrix
    """
    assert values.shape[-1] == 3
    result = np.full((*values.shape[:-1], 4, 4), np.identity(4), dtype=np.float32)
    result[..., 0:3, -1] = values
    return result


def decompose(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """decompose a transform matrix to translation, rotation and scale

    Args:
        transforms (..., 4, 4): transforms matrix

    Returns:
        Tuple[(..., 3), (..., 4), (..., 3)]: translation, rotation, scaling 
    """
    assert len(transforms.shape) >= 2 and transforms.shape[-1] == transforms.shape[-2] == 4

    translation = transforms[..., 0:3, 3]

    scaling_x = np.linalg.norm(transforms[..., 0:3, 0], axis=-1)
    scaling_y = np.linalg.norm(transforms[..., 0:3, 1], axis=-1)
    scaling_z = np.linalg.norm(transforms[..., 0:3, 2], axis=-1)
    scaling = np.stack([scaling_x, scaling_y, scaling_z], axis=-1, dtype=np.float32)

    mask = np.linalg.det(transforms)[..., np.newaxis]
    scaling = np.where(mask < 0, -scaling, scaling)
    rotation_matrix = transforms[..., 0:3, 0:3] / scaling[..., np.newaxis]
    quaternion: np.ndarray = Rotation.from_matrix(rotation_matrix.reshape((-1, 3, 3))).as_quat(True).reshape((*transforms.shape[:-2], 4))
    return translation.astype(transforms.dtype), quaternion.astype(transforms.dtype), scaling.astype(transforms.dtype)


def expand_rotation_matrix(rotations: np.ndarray):
    """Expand 3x3 rotation matrix to 4x4 transform matrix

    Args:
        rotations (..., 3, 3): 3d rotation matrix

    Returns:
        np.ndarray (..., 4, 4): 3d transform matrix 
    """
    assert len(rotations.shape) >= 2 and rotations.shape[-1] == rotations.shape[-2] == 3
    result = np.full((*rotations.shape[:-2], 4, 4), np.identity(4), dtype=np.float32)
    result[..., 0:3, 0:3] = rotations
    return result.astype(rotations.dtype)


def axis_angle_to_euler(order: str, axis_angles: np.ndarray):
    assert axis_angles.shape[-1] == 3
    rotations = Rotation.from_rotvec(axis_angles.reshape(-1, 3))
    return rotations.as_euler(order).reshape([*axis_angles.shape[:-1], -1]).astype(axis_angles.dtype)


def axis_angle_to_quat(axis_angles: np.ndarray):
    assert axis_angles.shape[-1] == 3
    rotations = Rotation.from_rotvec(axis_angles.reshape(-1, 3))
    return rotations.as_quat(True).reshape([*axis_angles.shape[:-1], 4]).astype(axis_angles.dtype)


def axis_angle_to_6d(axis_angles: np.ndarray):
    assert axis_angles.shape[-1] == 3
    rotations = Rotation.from_rotvec(axis_angles.reshape(-1, 3))
    forwards = rotations.apply(forward_direction)
    ups = rotations.apply(up_direction)
    return np.stack([forwards, ups]).reshape([*axis_angles.shape[:-1], 2, 3]).astype(axis_angles.dtype)


def quat_to_euler(order: str, quaternions: np.ndarray):
    assert quaternions.shape[-1] == 4
    rotations = Rotation.from_quat(quaternions.reshape((-1, 4)))
    return rotations.as_euler(order).reshape([*quaternions.shape[:-1], -1]).astype(quaternions.dtype)


def quat_to_axis_angle(quaternions: np.ndarray):
    assert quaternions.shape[-1] == 4
    rotations = Rotation.from_quat(quaternions.reshape((-1, 4)))
    return rotations.as_rotvec().reshape([*quaternions.shape[:-1], 3]).astype(quaternions.dtype)


def quat_to_matrix(quaternions: np.ndarray):
    assert quaternions.shape[-1] == 4
    rotations = Rotation.from_quat(quaternions.reshape((-1, 4)))
    return rotations.as_matrix().reshape([*quaternions.shape[:-1], 4, 4]).astype(quaternions.dtype)


def quat_to_6d(quaternions: np.ndarray):
    assert quaternions.shape[-1] == 4
    rotations = Rotation.from_quat(quaternions.reshape((-1, 4)))
    forwards = rotations.apply(forward_direction)
    ups = rotations.apply(up_direction)
    return np.stack([forwards, ups]).reshape([*quaternions.shape[:-1], 2, 3]).astype(quaternions.dtype)


def euler_to_quat(order: str, euler_angles: np.ndarray):
    assert euler_angles.shape[-1] == 3
    rotations = Rotation.from_euler(order, euler_angles.reshape((-1, 3)))
    return rotations.as_quat(True).reshape([*euler_angles.shape[:-1], 4]).astype(euler_angles.dtype)


def quat_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    assert src.shape == dst.shape and src.shape[-1] == 3

    v1: np.ndarray = src / np.linalg.norm(src, axis=-1, keepdims=True)
    v2: np.ndarray = dst / np.linalg.norm(dst, axis=-1, keepdims=True)

    v = np.cross(v1, v2)
    w = np.sqrt((v1**2).sum(axis=-1, keepdims=True) * (v2**2).sum(axis=-1, keepdims=True)) + (v1 * v2).sum(axis=-1, keepdims=True)

    q = np.concatenate([v, w], axis=-1)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)

    return q

def exchange_yz(vec:np.ndarray):
    result = vec.copy()
    result = result[..., [0, 2, 1]]
    result[..., 1] = - result[..., 1]
    return result