import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from mpl_toolkits import mplot3d


def get_axes_lims(dataset):
    """
    Given a dataset, finds same x-, y- and z-limits 3d plot.
    Works best for standardized datasets.
    """
    mins = []
    maxs = []
    for dim in range(dataset.shape[1]):
        min_ = np.min(dataset[:, dim])
        max_ = np.max(dataset[:, dim])
        mins.append(min_ * 1.2 if min_ < 0 else min_ / 1.2)
        maxs.append(max_ * 1.2 if max_ > 0 else max_ / 1.2)
    return mins + maxs


def get_axes_lims_two_datasets(dataset1, dataset2):
    """
    Given two datasets, finds same x-, y- and z-limits for both 3d plots
    that accomodate both datasets. Works best for standardized datasets.
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
    plot_time_evolution=False,
    title1="dataset1",
    title2="dataset2",
    plot=False,
    save=True,
    savepath=".",
    name=None,
    linewidth=0.5,
    labels=None,
    file_type=".jpg",
):
    """
    Creates two side-by-side 3D plots of two 3-dim. time-series.
    """
    if name is None:
        name = f"comparison_{title1}_{title2}".replace(" ", "_").lower()
    # adjust x, y and z to fit both datasets
    x_min, y_min, z_min, x_max, y_max, z_max = get_axes_lims_two_datasets(
        dataset1, dataset2
    )

    title1 = f"Time evolution: {title1}" if plot_time_evolution else title1
    title2 = f"Time evolution: {title2}" if plot_time_evolution else title2

    if labels is None:
        labels = ["x", "y", "z"]

    if not plot_time_evolution:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot3D(
            dataset1[:, 0], dataset1[:, 1], dataset1[:, 2], "blue", linewidth=linewidth
        )
        ax.set_title(title1, size=18)
        ax.scatter3D(
            *dataset1[0, :3], marker="o", color="green", s=50, label="initial state"
        )
        ax.scatter3D(
            *dataset1[-1, :3], marker="x", color="red", s=90, label="final state"
        )
        plt.legend(loc="upper right")
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        ax.set_zlim3d(z_min, z_max)
        ax.set_xlabel(labels[0], size=15)
        ax.set_ylabel(labels[1], size=15)
        ax.set_zlabel(labels[2], size=15)
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.plot3D(
            dataset2[:, 0], dataset2[:, 1], dataset2[:, 2], "blue", linewidth=linewidth
        )
        ax.set_title(title2, size=18)
        ax.scatter3D(
            *dataset2[0, :3], marker="o", color="green", s=50, label="initial state"
        )
        ax.scatter3D(
            *dataset2[-1, :3], marker="x", color="red", s=90, label="final state"
        )
        plt.legend(loc="upper right")
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        ax.set_zlim3d(z_min, z_max)
        ax.set_xlabel(labels[0], size=15)
        ax.set_ylabel(labels[1], size=15)
        ax.set_zlabel(labels[2], size=15)
    else:
        assert dataset1.shape == dataset2.shape
        tval = np.arange(dataset1.shape[0])
        fig = plt.figure(figsize=(25, 25))
        ax = fig.add_subplot(121, projection="3d")
        cax = ax.scatter(
            dataset1[:, 0],
            dataset1[:, 1],
            dataset1[:, 2],
            cmap=plt.cm.viridis,
            c=tval,
            s=15,
        )
        fig.colorbar(cax, shrink=0.3)
        ax.set_title(title1, size=18)
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        ax.set_zlim3d(z_min, z_max)
        ax.set_xlabel(labels[0], size=15)
        ax.set_ylabel(labels[1], size=15)
        ax.set_zlabel(labels[2], size=15)
        ax = fig.add_subplot(122, projection="3d")
        cax = ax.scatter(
            dataset2[:, 0],
            dataset2[:, 1],
            dataset2[:, 2],
            cmap=plt.cm.viridis,
            c=tval,
            s=15,
        )
        fig.colorbar(cax, shrink=0.3)
        ax.set_title(title2, size=18)
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        ax.set_zlim3d(z_min, z_max)
        ax.set_xlabel(labels[0], size=15)
        ax.set_ylabel(labels[1], size=15)
        ax.set_zlabel(labels[2], size=15)
    plt.subplots_adjust(wspace=-0.07, hspace=0)
    if plot:
        plt.show()
    if save:
        path = os.path.join(savepath, f"{name}{file_type}")
        plt.savefig(path)
        plt.close()


def plot_3d_dataset(
    dataset,
    plot_time_evolution=False,
    title="",
    plot=False,
    save=True,
    savepath=".",
    name=None,
    linewidth=0.5,
    ax_lims=None,
    labels=None,
    file_type=".jpg",
):
    """
    Creates 3D plot of 3-dim. time-series.
    """
    if ax_lims is None:
        x_min, y_min, z_min, x_max, y_max, z_max = get_axes_lims(dataset)
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = ax_lims
    if name is None:
        name = title.replace(" ", "_").lower()
    fig = plt.figure(figsize=(10, 10))

    if labels is None:
        labels = ["x", "y", "z"]

    if not plot_time_evolution:
        ax = plt.axes(projection="3d")
        ax.plot3D(
            dataset[:, 0], dataset[:, 1], dataset[:, 2], "blue", linewidth=linewidth
        )
        ax.scatter3D(
            *dataset[0, :3], marker="o", color="green", s=50, label="initial state"
        )
        ax.scatter3D(
            *dataset[-1, :3], marker="x", color="red", s=90, label="final state"
        )
    else:
        ax = fig.add_subplot(111, projection="3d")
        tval = np.arange(dataset.shape[0])
        cax = ax.scatter(
            dataset[:, 0],
            dataset[:, 1],
            dataset[:, 2],
            cmap=plt.cm.viridis,
            c=tval,
            s=15,
        )
        fig.colorbar(cax, shrink=0.6)
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)
    ax.set_xlabel(labels[0], size=15)
    ax.set_ylabel(labels[1], size=15)
    ax.set_zlabel(labels[2], size=15)
    title_ = f"Time evolution: {title}" if plot_time_evolution else title
    plt.title(title_, size=18)
    plt.legend(loc="upper right")
    plt.draw()
    if plot:
        plt.show()
    if save:
        path = os.path.join(savepath, f"{name}{file_type}")
        plt.savefig(path)
        plt.close()


def get_state_space_volume(data):
    """
    Returns volume of the smallest cube filled by a 3D dataset.
    """
    if not data.shape[1] == 3:
        raise Exception("Dataset must be three-dimensional!")
    ranges = np.abs(data.max(axis=0) - data.min(axis=0))
    return ranges[0] * ranges[1] * ranges[2]


def has_more_state_space_volume(data1, data2):
    """
    Checks if data1 covers a larger portion of the latent space than data2.
    """
    vol1 = get_state_space_volume(data1)
    vol2 = get_state_space_volume(data2)
    return vol1 > vol2


def compare_3d_datasets_one_plot(
    dataset1,
    dataset2,
    title="",
    label1="dataset1",
    label2="dataset2",
    plot=False,
    save=True,
    savepath=".",
    name=None,
    linewidth=0.5,
    ax_lims=None,
    labels=None,
    file_type=".jpg",
    alpha=1.0,
):
    """
    Creates 3D plot of two 3-dim. time-series.
    """
    # adjust x, y and z to fit both datasets
    if ax_lims is None:
        x_min, y_min, z_min, x_max, y_max, z_max = get_axes_lims_two_datasets(
            dataset1, dataset2
        )
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = ax_lims
    if name is None:
        name = f"comparison_{label1}_{label2}".replace(" ", "_").lower()

    if labels is None:
        labels = ["x", "y", "z"]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")

    # plot that trajectory first which fills a larger part
    # of the state space, for better visibility
    if has_more_state_space_volume(dataset1, dataset2):
        ax.plot3D(
            dataset1[:, 0],
            dataset1[:, 1],
            dataset1[:, 2],
            "blue",
            linewidth=linewidth,
            label=label1,
            alpha=alpha,
        )
        ax.plot3D(
            dataset2[:, 0],
            dataset2[:, 1],
            dataset2[:, 2],
            "orange",
            linewidth=linewidth,
            label=label2,
            alpha=alpha,
        )
    else:
        ax.plot3D(
            dataset2[:, 0],
            dataset2[:, 1],
            dataset2[:, 2],
            "orange",
            linewidth=linewidth,
            label=label2,
            alpha=alpha,
        )
        ax.plot3D(
            dataset1[:, 0],
            dataset1[:, 1],
            dataset1[:, 2],
            "blue",
            linewidth=linewidth,
            label=label1,
            alpha=alpha,
        )
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_min, y_max)
    ax.set_zlim3d(z_min, z_max)
    ax.set_xlabel(labels[0], size=15)
    ax.set_ylabel(labels[1], size=15)
    ax.set_zlabel(labels[2], size=15)
    plt.title(title, size=18)
    plt.legend(loc="upper right")
    if plot:
        plt.show()
    if save:
        path = os.path.join(savepath, f"{name}{file_type}")
        plt.savefig(path)
        plt.close()


def plot_time_series(
    dataset,
    title="",
    plot=False,
    save=True,
    savepath=".",
    name=None,
    labels=None,
    file_type=".jpg",
):
    """
    Plots every dimension of a dataset separately.
    """
    if name is None:
        name = title.replace(" ", "_").lower()
    n = dataset.shape[1]
    fig, ax = plt.subplots(n, 1, figsize=(15, n * 4))

    if labels is None:
        labels = [f"$X_{i}$" for i in range(1, n + 1)]

    for i in range(n):
        ax[i].plot(dataset[:, i])
        ax[i].set_xlabel("t", size=15)
        ax[i].set_ylabel(labels[i], size=15)
    plt.suptitle(title)
    if plot:
        plt.show()
    if save:
        path = os.path.join(savepath, f"{name}{file_type}")
        plt.savefig(path)
        plt.close()


def compare_time_series(
    dataset1,
    dataset2,
    title="",
    label1="dataset1",
    label2="dataset2",
    plot=False,
    save=True,
    savepath=".",
    name=None,
    labels=None,
    file_type=".jpg",
    line_idx=None,
):
    """
    Plots every dimension of a comparison of two datasets separately.
    """
    assert dataset1.shape == dataset2.shape
    if name is None:
        name = f"time_series_comparison_{label1}_{label2}".replace(" ", "_").lower()
    n = dataset1.shape[1]
    fig, ax = plt.subplots(n, 1, figsize=(15, n * 4))

    if labels is None:
        labels = [f"$X_{i}$" for i in range(1, n + 1)]

    for i in range(n):
        ax[i].plot(dataset1[:, i], label=label1)
        ax[i].plot(dataset2[:, i], label=label2)
        ax[i].set_xlabel("t", size=15)
        ax[i].set_ylabel(labels[i], size=15)
        if not line_idx is None:
            ymin, ymax = ax[i].get_ylim()
            ax[i].vlines(
                line_idx,
                ymin,
                ymax,
                linestyle="--",
                color="gray",
                linewidths=2.5,
                label="Start of pw. gen. data",
            )
            ax[i].set_ylim(top=ymax, bottom=ymin)
        ax[i].legend(loc="upper right")

    plt.suptitle(title, size=18)
    if plot:
        plt.show()
    if save:
        path = os.path.join(savepath, f"{name}{file_type}")
        plt.savefig(path)
        plt.close()
