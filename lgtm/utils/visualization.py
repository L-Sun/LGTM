from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


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
