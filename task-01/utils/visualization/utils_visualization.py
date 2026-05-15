import numpy as np
import matplotlib.pyplot as plt

def barplot_mean_sem(
    X,
    d,
    bar_labels=None,
    ax=None,
    title=None,
    ylabel=None,
    xlabel=None,
):
    """
    Plot mean Â± SEM of a matrix/ndarray along dimension d.

    Parameters
    ----------
    X : array-like
        Input data.
    d : int
        Dimension along which to compute mean and SEM.
        d=0 means compute across rows,
        d=1 means compute across columns.
    bar_labels : list of str, optional
        Labels for the bars.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, a new figure/axis is created.
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    xlabel : str, optional
        X-axis label.

    Returns
    -------
    means : np.ndarray
        Mean values.
    sem : np.ndarray
        Standard error of the mean.
    ax : matplotlib.axes.Axes
        Axis containing the plot.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim < 1:
        raise ValueError("X must have at least 1 dimension.")

    if d not in [0, 1]:
        raise ValueError(f"d must be 0 or 1, got {d}")

    if d >= X.ndim:
        raise ValueError(f"Invalid axis d={d} for X with {X.ndim} dimensions.")

    means = np.nanmean(X, axis=d)
    n = np.sum(~np.isnan(X), axis=d)
    std = np.nanstd(X, axis=d, ddof=1)
    sem = std / np.sqrt(n)

    means = np.ravel(means)
    sem = np.ravel(sem)

    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(len(means))
    ax.bar(x, means, yerr=sem, capsize=5)
    ax.set_xticks(x)

    if bar_labels is not None:
        if len(bar_labels) != len(means):
            raise ValueError(f"bar_labels has length {len(bar_labels)}, "f"but there are {len(means)} bars.")
        ax.set_xticklabels(bar_labels)
    else:
        ax.set_xticklabels([str(i + 1) for i in range(len(means))])

    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    return means, sem, ax



def scatter_value_map(
    coords,
    values,
    ax=None,
    cmap="viridis",
    s=40,
    title=None,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    colorbar_label=None,
):
    """
    Plot a 2D or 3D scatter where point color represents a value.

    Parameters
    ----------
    coords : array-like, shape (N, 2) or (N, 3)
        Coordinates of the data points.
    values : array-like, shape (N,)
        Scalar value for each point. Used for color mapping.
    ax : matplotlib axis, optional
        Existing axis to plot on. If None, a new figure/axis is created.
    cmap : str, optional
        Matplotlib colormap name.
    s : float, optional
        Marker size.
    title : str, optional
        Plot title.
    xlabel, ylabel, zlabel : str, optional
        Axis labels.
    colorbar_label : str, optional
        Label for the colorbar.

    Returns
    -------
    scatter : PathCollection
        The scatter object.
    ax : matplotlib axis
        The axis used for plotting.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).ravel()

    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array of shape (N, 2) or (N, 3).")

    n_points, n_dims = coords.shape

    if n_dims not in (2, 3):
        raise ValueError("coords must have 2 or 3 columns.")

    if len(values) != n_points:
        raise ValueError(f"values must have length {n_points}, but got {len(values)}.")

    if ax is None:
        fig = plt.figure()
        if n_dims == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    if n_dims == 2:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=cmap,
            s=s,
            marker="o",
        )
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    else:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=values,
            cmap=cmap,
            s=s,
            marker="o",
            depthshade=True,
        )
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if zlabel is not None:
            ax.set_zlabel(zlabel)

    if title is not None:
        ax.set_title(title)

    cbar = plt.colorbar(scatter, ax=ax)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)

    return scatter, ax



def plot_heatmap(
    mat,
    row_labels=None,
    col_labels=None,
    ax=None,
    cmap="bwr",
    center_zero=True,
    vmin=None,
    vmax=None,
    title=None,
    xlabel=None,
    ylabel=None,
    colorbar_label=None,
    rotation=45,
):
    """
    Plot a matrix as a heatmap.

    Parameters
    ----------
    mat : array-like, shape (M, N)
        Matrix to plot.
    row_labels : list of str, optional
        Labels for rows.
    col_labels : list of str, optional
        Labels for columns.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If None, a new figure/axis is created.
    cmap : str, optional
        Matplotlib colormap name.
    center_zero : bool, optional
        If True and vmin/vmax are not given, use symmetric limits around zero.
    vmin, vmax : float, optional
        Color scale limits.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    colorbar_label : str, optional
        Label for the colorbar.
    rotation : float, optional
        Rotation angle for x tick labels.

    Returns
    -------
    im : AxesImage
        The image object.
    ax : matplotlib.axes.Axes
        Axis containing the plot.
    """
    mat = np.asarray(mat, dtype=float)

    if mat.ndim != 2:
        raise ValueError("mat must be a 2D array.")

    n_rows, n_cols = mat.shape

    if row_labels is not None and len(row_labels) != n_rows:
        raise ValueError(f"row_labels has length {len(row_labels)}, but mat has {n_rows} rows.")

    if col_labels is not None and len(col_labels) != n_cols:
        raise ValueError(f"col_labels has length {len(col_labels)}, but mat has {n_cols} columns.")

    if ax is None:
        _, ax = plt.subplots()

    if center_zero and vmin is None and vmax is None:
        m = np.nanmax(np.abs(mat))
        vmin = -m
        vmax = m

    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))

    if col_labels is not None:
        ax.set_xticklabels(col_labels, rotation=rotation, ha="right")
    else:
        ax.set_xticklabels([str(i + 1) for i in range(n_cols)], rotation=rotation, ha="right")

    if row_labels is not None:
        ax.set_yticklabels(row_labels)
    else:
        ax.set_yticklabels([str(i + 1) for i in range(n_rows)])

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    cbar = plt.colorbar(im, ax=ax)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)

    return im, ax