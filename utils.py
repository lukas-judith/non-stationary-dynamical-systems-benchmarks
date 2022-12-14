import pdb
import os
import numpy as np
import matplotlib.pyplot as plt


def standardize(data):
    """
    Standardizes input data.
    """
    data_stand = (data - np.mean(data, 0)) / np.std(data, 0)
    return data_stand


def standardize_two_datasets(data1, data2):
    """
    Standardize two datasets with same mean and std.
    Both are chosen from first dataset.
    """
    assert data1.shape == data2.shape
    mean = np.mean(data1, 0)
    std = np.std(data1, 0)
    data1_stand = (data1 - mean) / std
    data2_stand = (data2 - mean) / std
    return data1_stand, data2_stand


def standardize_several_datasets(datasets):
    """
    Standardize multiple datasets with same mean and std.
    Both are chosen from first dataset.
    """
    datasets_stand = []
    data1 = datasets[0]
    mean = np.mean(data1, 0)
    std = np.std(data1, 0)
    for i, d in enumerate(datasets):
        d_stand = (d - mean) / std
        datasets_stand.append(d_stand)
    return datasets_stand


def find_R0_from_fixed_point(FP, R1):
    """
    Given a desired fixed point FP and parameter R1 of AR1 model,
    finds R0 of AR1 model.
    """
    R0 = (np.eye(R1.shape[0]) - R1) @ FP
    return R0


def find_R0_with_R1_unity(param0, param_final, n):
    """
    For R1 = identity, determine the R0 needed to reach param_final
    after n iterations of 'param_new = param_old + R0'.
    """
    R0 = (param_final - param0) / n
    R1 = np.eye(R0.shape[0])
    return R0, R1


def get_stationary_AR1_params(model, dim_z=None):
    """
    Returns AR1 parameters such that the resulting AR1 model leaves
    the initial parameters unchanged.
    """
    if model == "PLRNN":
        R0_AW = np.zeros((dim_z, dim_z))
        R0_h = np.zeros((dim_z,))
        R1_AW = np.eye(dim_z)
        R1_h = np.eye(dim_z)
        params = [R0_AW, R1_AW, R0_h, R1_h]
    elif model == "L63":
        R0 = np.zeros((3,))
        R1 = np.eye(3)
        params = [R0, R1]
    else:
        raise Exception(f"No known model {model}!")
    return params


def get_max_eigenvalue(M):
    """
    Returns maximum (absolute) eigenvalue of square matrix M.
    """
    eigenvalues = np.linalg.eigvals(M)
    max_abs_eig = np.max(np.abs(eigenvalues))
    return max_abs_eig


def iter_linear_map(M, C, X0, n):
    """
    Iterates map 'X_new = M @ X_old + C' for n times."""
    # X0 can be vector or matrix; make sure it has correct shape
    out = X0.reshape(C.shape)
    for t in range(n):
        out = M @ out + C
    return out


def compute_fixed_point_linear_map(M, C):
    """
    Computes fixed point for multivariate affine map.
    """
    return np.linalg.inv(np.eye(M.shape[0]) - M) @ C


def has_converged_to_fixed_point_linear(AW, R0, R1, n):
    """
    Checks whether the linear map 'AW_new = R1 @ AW_old + R0' has converged
    to a fixed point after n iterations.
    """
    AW_n = iter_linear_map(R1, R0, AW, n)
    FP = compute_fixed_point_linear_map(R1, R0)
    return np.allclose(FP, AW_n)


def get_axes_lims(dataset1, dataset2):
    """
    Given two datasets, finds same x-, y- and z-limits for both 3d plots
    that accomodate both datasets.
    """
    assert dataset1.shape[1] == dataset2.shape[1]
    mins = []
    maxs = []
    for dim in range(dataset1.shape[1]):
        min_ = np.min((dataset1[:, dim], dataset2[:, dim]))
        max_ = np.max((dataset1[:, dim], dataset2[:, dim]))
        mins.append(min_ * 1.2 if min_ < 0 else min_ / 1.2)
        maxs.append(max_ * 1.2 if max_ > 0 else max_ / 1.2)
    return mins + maxs


def compare_3d_datasets(
    dataset1,
    dataset2,
    title1="dataset1",
    title2="dataset2",
    plot=True,
    save=False,
    savepath=".",
    name=None,
):
    """
    Creates two side-by-side 3D plots of two 3-dim. time-series.
    """
    if name is None:
        name = f"comparison_{title1}_{title2}"
    linewidth1 = 10000 / len(np.unique(np.round(dataset1, 3), axis=0))
    linewidth2 = 10000 / len(np.unique(np.round(dataset2, 3), axis=0))
    # maximum linewidth set to 1
    linewidth1 = 1 if linewidth1 > 1 else linewidth1
    linewidth2 = 1 if linewidth2 > 1 else linewidth2
    # adjust x, y and z to fit both datasets
    x_min, y_min, z_min, x_max, y_max, z_max = get_axes_lims(dataset1, dataset2)

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot3D(
        dataset1[:, 0], dataset1[:, 1], dataset1[:, 2], "blue", linewidth=linewidth1
    )
    ax.set_title(title1)
    ax.scatter3D(
        *dataset1[0, :3], marker="o", color="green", s=50, label="initial state"
    )
    ax.scatter3D(*dataset1[-1, :3], marker="x", color="red", s=90, label="final state")
    plt.legend()
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot3D(
        dataset2[:, 0], dataset2[:, 1], dataset2[:, 2], "blue", linewidth=linewidth2
    )
    ax.set_title(title2)
    ax.scatter3D(
        *dataset2[0, :3], marker="o", color="green", s=50, label="initial state"
    )
    ax.scatter3D(*dataset2[-1, :3], marker="x", color="red", s=90, label="final state")
    plt.legend()
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)
    if plot:
        plt.show()
    if save:
        path = os.path.join(savepath, f"{name}.pdf")
        plt.savefig(path)


def plot_3d_dataset(dataset, name="test", plot=True, save=False, savepath="."):
    """
    Creates 3D plot of 3-dim. time-series.
    """
    linewidth = 10000 / len(np.unique(np.round(dataset, 3), axis=0))
    # maximum linewidth set to 1
    linewidth = 1 if linewidth > 1 else linewidth

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.plot3D(dataset[:, 0], dataset[:, 1], dataset[:, 2], "blue", linewidth=linewidth)
    ax.scatter3D(
        *dataset[0, :3], marker="o", color="green", s=50, label="initial state"
    )
    ax.scatter3D(*dataset[-1, :3], marker="x", color="red", s=90, label="final state")
    plt.title(name.replace("_", " "))
    plt.legend()
    plt.draw()
    if plot:
        plt.show()
    if save:
        path = os.path.join(savepath, f"{name}.pdf")
        plt.savefig(path)
