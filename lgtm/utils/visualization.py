from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import torch
from third_packages.fmbvh.motion_tensor.rotations import get_quat_from_pos
from third_packages.fmbvh.motion_tensor.bvh_casting import write_offsets_to_bvh, write_quaternion_to_bvh
from third_packages.fmbvh.bvh.editor import build_bvh_from_scratch
import json
import torch
import numpy as np


def calc_axis_lim(positions: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    x_min = positions[..., 0].min()
    x_max = positions[..., 0].max()

    y_min = positions[..., 1].min()
    y_max = positions[..., 1].max()

    z_min = positions[..., 2].min()
    z_max = positions[..., 2].max()

    lim_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max()
    x_min = 0.5 * (x_max + x_min) - 0.5 * lim_range
    x_max = 0.5 * (x_max + x_min) + 0.5 * lim_range

    y_min = 0.5 * (y_max + y_min) - 0.5 * lim_range
    y_max = 0.5 * (y_max + y_min) + 0.5 * lim_range

    z_min = z_min
    z_max = z_min + lim_range

    return (x_min, x_max), (y_min, y_max), (z_min, z_max)


def plot_pose(positions: np.ndarray, joint_parents: np.ndarray):
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"the shape of position must follow (num_joints, 3) but got {positions.shape}")

    if not np.issubdtype(joint_parents.dtype, np.integer):
        raise ValueError(f"the expected dtype of joint_parents is np.integer but got {joint_parents.dtype}!")

    if joint_parents.ndim != 1:
        raise ValueError(f"joint_parents must be 1D array but its ndim={joint_parents.ndim}")

    if positions.shape[0] != len(joint_parents):
        raise ValueError(f"the positions({positions.shape}) must match number of joints({joint_parents.shape[0]})")

    ax: Axes
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    lim = calc_axis_lim(positions)
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_zlim(*lim[2]) # type: ignore

    for joint, parent in enumerate(joint_parents):
        if joint == 0:
            continue
        ax.plot(
            [positions[joint, 0], positions[parent, 0]],
            [positions[joint, 1], positions[parent, 1]],
            [positions[joint, 2], positions[parent, 2]],
            "o-",
        )

    plt.close()

    return fig


def animate(positions: np.ndarray, joint_parents: np.ndarray, fps: float, title=""):
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"the shape of position must follow (num_frames, num_joints, 3) but got {positions.shape}")

    if joint_parents.ndim != 1:
        raise ValueError(f"the joint_parents must be 1D Array, but its ndim={joint_parents.ndim}!")

    if not np.issubdtype(joint_parents.dtype, np.integer):
        raise ValueError(f"the expected dtype of joint_parents is np.integer but got {joint_parents.dtype}!")

    if joint_parents.ndim != 1:
        raise ValueError(f"joint_parents must be 1D array but its ndim={joint_parents.ndim}")

    if positions.shape[1] != len(joint_parents):
        raise ValueError(f"the positions.shape[2] must match number of joints({joint_parents.shape[0]})")

    import matplotlib.animation as animation

    lim = calc_axis_lim(positions)

    ax: Axes3D
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    lines = [ax.plot([], [], [], "o-")[0] for _ in range(joint_parents.shape[0])]
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_zlim(*lim[2])

    ax.text2D(0.0, 0.7, title, transform=ax.transAxes)

    # resample to 60 fps
    from lgtm.utils.data_processing import resample_motion_features
    positions = resample_motion_features(positions, fps, 60)

    def tick(frame):
        for joint, parent in enumerate(joint_parents):
            if joint == 0:
                continue
            line = [
                np.array([positions[frame, joint, 0], positions[frame, parent, 0]]),
                np.array([positions[frame, joint, 1], positions[frame, parent, 1]]),
                np.array([positions[frame, joint, 2], positions[frame, parent, 2]]),
            ]
            lines[joint].set_data(line[0], line[1])
            lines[joint].set_3d_properties(line[2]) # type: ignore

        return lines

    ani = animation.FuncAnimation(fig, tick, positions.shape[0], interval=1000 / 60)
    plt.close()
    return ani


# recover quats from positions and save to bvh mocap file
def pos_to_bvh(positions, output_path=None, x=True, m=False, mm=False):
    """
    :param positions: [F, J, 3]  #  frames, joints, 3
    :param output_path: output bvh file path
    :param x: flip x-axis
    :param m: mirror the body part by swapping them
    :param mm: 
    """
    t_json = """
    [[[0, 0, 0],
    [-0.0561437, -0.09454167, -0.02347454],
    [0.05786965, -0.1051669, -0.01655883],
    [0.001336131, 0.1104168, -0.03792468],
    [-0.06722913, -0.3968683, -0.006654377],
    [0.0507268, -0.3798212, -0.01445162],
    [-0.01020924, 0.1509713, 0.00444234],
    [0.04559392, -0.4212858, -0.04114642],
    [-0.01715488, -0.434912, -0.03993758],
    [0.008991977, 0.05784686, 0.0226697],
    [-0.04430889, -0.06073571, 0.1351566],
    [0.03544413, -0.05966973, 0.1418128],
    [0.009582404, 0.1660209, -0.02724041],
    [-0.04242062, 0.07627739, -0.005254913],
    [0.0460769, 0.07681628, -0.008529007],
    [-0.02290352, 0.1607197, 0.02293979],
    [-0.1409457, 0.06041686, -0.01489903],
    [0.1303553, 0.05884924, -0.01258629],
    [-0.2547487, -0.07616559, -0.04549227],
    [0.2724926, -0.0459256, -0.03242894],
    [-0.2716039, 0.0256131, -0.000648317],
    [0.2633793, 2.866859E-05, -0.01182221]]]
    """

    def to_np(*x):
        return tuple([e.numpy() for e in x]) if len(x) > 1 else x[0].numpy()


    def to_th(*x):
        return tuple([torch.from_numpy(e) for e in x]) if len(x) > 1 else torch.from_numpy(x[0])

    p_index = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    mirrored = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]
    
    target = positions
    target = np.transpose(target, (1, 2, 0))  # [F, J, 3] -> [J, 3, F]

    offset = json.loads(t_json)
    offset = np.array(offset)  # [1, 22, 3]
    offset = np.transpose(offset, (1, 2, 0))  # [F, J, 3] -> [J, 3, F]

    target, offset = to_th(target, offset)
    target = target.to(float)
    offset = offset.to(float)

    # --- fix --- #
    if x:
        target[:, 0, :].neg_()
    if m:
        for ia, ib in mirrored:
            target[[ia, ib]] = target[[ib, ia]]
    if mm:
        for ia, ib in mirrored:
            bb = target[[ib]]
            bb[:, 0, :].neg_()
            target[[ia]] = bb
    # ----------- # 

    obj = build_bvh_from_scratch(p_index, [f'J{e}' for e in range(len(p_index))], 20)
    src = [f'J{e}' for e in range(len(p_index))]
    dst = obj.names
    ls_src = [i for i in range(len(p_index))]
    to_dst = [src.index(e) for e in dst]
    to_src = [dst.index(e) for e in src]
    target = target[to_dst]
    offset = offset[to_dst]
    trans, quats = get_quat_from_pos(obj.p_index, target, offset)
    obj = write_offsets_to_bvh(offset, obj)
    obj = write_quaternion_to_bvh(trans, quats, obj)
    if output_path is not None:
        obj.to_file(output_path)
    return obj
