# plot_vector_field.py
"""
Utilities for visualising 3D vector fields built from vampyr FunctionTrees.

Two backends are provided:
  - matplotlib : lightweight, good for quick 2D slices
  - plotly     : interactive 3D cone plot (recommended for full 3D inspection)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401  (registers 3d projection)
from vampyr import vampyr3d as vp
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# Grid sampling
# ---------------------------------------------------------------------------

def sample_tree_on_grid(
    tree: vp.FunctionTree,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a real FunctionTree on a regular Cartesian grid.

    Parameters
    ----------
    tree : vp.FunctionTree
        The scalar field to sample.
    xs, ys, zs : 1-D arrays
        Coordinate vectors along each axis.

    Returns
    -------
    out : ndarray of shape (len(xs), len(ys), len(zs))
        Pointwise values of `tree`.
    """
    out = np.zeros((len(xs), len(ys), len(zs)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                out[i, j, k] = tree([x, y, z])
    return out


def make_grid(box: float, n: int = 10):
    """
    Build a uniform 3-D grid inside [-box, box]^3.

    Returns (xs, ys, zs, X, Y, Z) where X/Y/Z are meshgrid arrays.
    Avoids the box boundary (offset by 0.05*box) to prevent tree-edge artefacts.
    """
    offset = 0.05 * box
    coords = np.linspace(-box + offset, box - offset, n)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    return coords, coords, coords, X, Y, Z


# ---------------------------------------------------------------------------
# Main visualisation entry points
# ---------------------------------------------------------------------------

def plot_quiver_slice(
    Fx: vp.FunctionTree,
    Fy: vp.FunctionTree,
    Fz: vp.FunctionTree,
    box: float,
    n: int = 20,
    slice_axis: str = 'z',
    slice_val: float = 0.0,
    normalize: bool = True,
    title: str = "Vector field slice",
) -> None:
    """
    Plot a 2-D quiver slice through the vector field.

    Parameters
    ----------
    Fx, Fy, Fz : FunctionTree
        x-, y-, z-components of the vector field.
    box : float
        Half-width of the MRA box.
    n : int
        Grid resolution per axis in the slice plane.
    slice_axis : 'x', 'y', or 'z'
        Normal direction of the cutting plane.
    slice_val : float
        Position of the cutting plane along `slice_axis`.
    normalize : bool
        If True, all arrows have equal length (direction only).
    title : str
        Plot title.
    """
    offset = 0.05 * box
    coords = np.linspace(-box + offset, box - offset, n)

    # Build 2-D sample points in the slice plane
    U = np.zeros((n, n))
    V = np.zeros((n, n))

    if slice_axis == 'z':
        A, B = np.meshgrid(coords, coords, indexing='ij')
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                U[i, j] = Fx([a, b, slice_val])
                V[i, j] = Fy([a, b, slice_val])
        xlabel, ylabel = 'x', 'y'
    elif slice_axis == 'y':
        A, B = np.meshgrid(coords, coords, indexing='ij')
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                U[i, j] = Fx([a, slice_val, b])
                V[i, j] = Fz([a, slice_val, b])
        xlabel, ylabel = 'x', 'z'
    else:  # 'x'
        A, B = np.meshgrid(coords, coords, indexing='ij')
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                U[i, j] = Fy([slice_val, a, b])
                V[i, j] = Fz([slice_val, a, b])
        xlabel, ylabel = 'y', 'z'

    if normalize:
        mag = np.sqrt(U**2 + V**2) + 1e-30
        U, V = U / mag, V / mag

    plt.figure(figsize=(7, 6))
    plt.quiver(A, B, U, V, np.sqrt(U**2 + V**2), cmap='viridis')
    plt.colorbar(label='|F| (arb.)' if normalize else '|F|')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}  [{slice_axis}={slice_val:.2f}]")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()





def plot_cone_3d(
    Fx: vp.FunctionTree,
    Fy: vp.FunctionTree,
    Fz: vp.FunctionTree,
    box: float,
    n: int = 8,
    normalize: bool = False,
    title: str = "3D vector field",
) -> None:
    """
    Interactive 3-D cone plot using Plotly.

    Each cone points along (Fx, Fy, Fz) evaluated at that grid node.
    Requires: ``pip install plotly``

    Parameters
    ----------
    n : int
        Grid resolution per axis (keep ≤ 12 for readability; n^3 cones total).
    normalize : bool
        If True, rescale all cones to unit length (shows direction only).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Install plotly:  pip install plotly")

    xs, ys, zs, X, Y, Z = make_grid(box, n)

    print(f"Sampling {n**3} grid points for each component...")
    Ux = sample_tree_on_grid(Fx, xs, ys, zs)
    Uy = sample_tree_on_grid(Fy, xs, ys, zs)
    Uz = sample_tree_on_grid(Fz, xs, ys, zs)

    if normalize:
        mag = np.sqrt(Ux**2 + Uy**2 + Uz**2) + 1e-30
        Ux, Uy, Uz = Ux / mag, Uy / mag, Uz / mag

    fig = go.Figure(data=go.Cone(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        u=Ux.ravel(), v=Uy.ravel(), w=Uz.ravel(),
        colorscale='Viridis',
        sizemode='scaled',
        sizeref=0.5,
        colorbar=dict(title='|F|'),
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x (a.u.)',
            yaxis_title='y (a.u.)',
            zaxis_title='z (a.u.)',
        )
    )
    fig.show()


def plot_quiver_3d_matplotlib(
    Fx: vp.FunctionTree,
    Fy: vp.FunctionTree,
    Fz: vp.FunctionTree,
    box: float,
    n: int = 6,
    normalize: bool = True,
    title: str = "3D vector field",
) -> None:
    """
    Static 3-D quiver plot using matplotlib (no extra dependencies).

    Keep n ≤ 8; matplotlib's quiver3D gets cluttered quickly.
    Arrow colour encodes the local vector magnitude.
    """
    import matplotlib.cm as cm

    xs, ys, zs, X, Y, Z = make_grid(box, n)

    print(f"Sampling {n**3} grid points for each component...")
    Ux = sample_tree_on_grid(Fx, xs, ys, zs)
    Uy = sample_tree_on_grid(Fy, xs, ys, zs)
    Uz = sample_tree_on_grid(Fz, xs, ys, zs)

    mag = np.sqrt(Ux**2 + Uy**2 + Uz**2)

    if normalize:
        Ux, Uy, Uz = Ux / (mag + 1e-30), Uy / (mag + 1e-30), Uz / (mag + 1e-30)

    # Build per-arrow colours from the original magnitude
    cmap = plt.viridis
    norm = Normalize(vmin=mag.min(), vmax=mag.max())
    colors = cmap(norm(mag.ravel()))

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, (xi, yi, zi, ui, vi, wi) in enumerate(zip(
        X.ravel(), Y.ravel(), Z.ravel(),
        Ux.ravel(), Uy.ravel(), Uz.ravel(),
    )):
        ax.quiver(xi, yi, zi, ui, vi, wi,
                  length=0.15 * box, normalize=False,
                  arrow_length_ratio=0.4, color=colors[i])

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='|F| (a.u.)')

    ax.set_xlabel('x (a.u.)')
    ax.set_ylabel('y (a.u.)')
    ax.set_zlabel('z (a.u.)')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()






# ---------------------------------------------------------------------------
# Scalar field visualisation
# ---------------------------------------------------------------------------

def plot_scalar_slice(
    tree: vp.FunctionTree,
    box: float,
    n: int = 80,
    slice_axis: str = 'z',
    slice_val: float = 0.0,
    cmap: str = 'RdBu_r',
    log_scale: bool = False,
    contours: bool = False,
    n_contours: int = 20,
    title: str = "Scalar field slice",
    vmin: float = None,
    vmax: float = None,
) -> None:
    """
    Plot a 2-D heatmap (or contour) slice through a scalar FunctionTree.

    Parameters
    ----------
    tree : vp.FunctionTree
        The real scalar field to visualise (e.g. a density, potential, or
        a single ClifFunc component).
    box : float
        Half-width of the MRA box in atomic units.
    n : int
        Number of sample points per axis in the slice plane. Because this
        is a 2-D sample, n=80 is cheap (~6400 tree evaluations).
    slice_axis : 'x', 'y', or 'z'
        Normal direction of the cutting plane.
    slice_val : float
        Position of the cut along `slice_axis` in atomic units.
    cmap : str
        Matplotlib colormap. Use 'RdBu_r' for signed fields (densities,
        potentials), 'viridis' for positive-definite ones.
    log_scale : bool
        If True, plot log10(|field|). Useful for densities spanning many
        orders of magnitude near the nucleus.
    contours : bool
        If True, overlay contour lines on top of the heatmap.
    n_contours : int
        Number of contour levels when `contours=True`.
    title : str
        Plot title.
    vmin, vmax : float, optional
        Color axis limits. If None, auto-scaled symmetrically for signed
        fields or from data range for positive fields.
    """
    offset = 0.05 * box
    coords = np.linspace(-box + offset, box - offset, n)
    F = np.zeros((n, n))

    if slice_axis == 'z':
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                F[i, j] = tree([a, b, slice_val])
        xlabel, ylabel = 'x (a.u.)', 'y (a.u.)'
        A, B = np.meshgrid(coords, coords, indexing='ij')
    elif slice_axis == 'y':
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                F[i, j] = tree([a, slice_val, b])
        xlabel, ylabel = 'x (a.u.)', 'z (a.u.)'
        A, B = np.meshgrid(coords, coords, indexing='ij')
    else:  # 'x'
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                F[i, j] = tree([slice_val, a, b])
        xlabel, ylabel = 'y (a.u.)', 'z (a.u.)'
        A, B = np.meshgrid(coords, coords, indexing='ij')

    plot_data = np.log10(np.abs(F) + 1e-30) if log_scale else F

    # Auto symmetric color limits for signed data
    if vmin is None and vmax is None and not log_scale:
        absmax = np.max(np.abs(plot_data))
        vmin, vmax = -absmax, absmax

    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(A, B, plot_data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('log₁₀|f| (a.u.)' if log_scale else 'f (a.u.)')

    if contours:
        levels = np.linspace(np.min(plot_data), np.max(plot_data), n_contours)
        ax.contour(A, B, plot_data, levels=levels, colors='k', linewidths=0.4, alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  [{slice_axis}={slice_val:.2f} a.u.]")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_scalar_slice_3D(
    tree: vp.FunctionTree,
    box: float,
    n: int = 80,
    slice_axis: str = 'z',
    slice_val: float = 0.0,
    cmap: str = 'RdBu_r',
    log_scale: bool = False,
    title: str = "Scalar field slice (3D)",
    vmin: float = None,
    vmax: float = None,
    elev: float = 30,
    azim: float = -60,
    edgecolor: str = 'none',
    linewidth: float = 0.0,
) -> None:
    """
    Plot a planar slice of a FunctionTree as a 3D surface.

    The x/y coordinates correspond to the slice coordinates while the
    surface height is given by the scalar field value. The surface is
    additionally colored according to the local value.
    """

    offset = 0.05 * box
    coords = np.linspace(-box + offset, box - offset, n)

    F = np.zeros((n, n))

    if slice_axis == 'z':
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                F[i, j] = tree([a, b, slice_val])
        xlabel, ylabel = 'x (a.u.)', 'y (a.u.)'

    elif slice_axis == 'y':
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                F[i, j] = tree([a, slice_val, b])
        xlabel, ylabel = 'x (a.u.)', 'z (a.u.)'

    else:  # x
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                F[i, j] = tree([slice_val, a, b])
        xlabel, ylabel = 'y (a.u.)', 'z (a.u.)'

    A, B = np.meshgrid(coords, coords, indexing='ij')

    plot_data = np.log10(np.abs(F) + 1e-30) if log_scale else F

    if vmin is None and vmax is None:
        if log_scale:
            vmin = np.min(plot_data)
            vmax = np.max(plot_data)
        else:
            absmax = np.max(np.abs(plot_data))
            vmin, vmax = -absmax, absmax

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        A,
        B,
        plot_data,
        facecolors=cmap_obj(norm(plot_data)),
        rstride=1,
        cstride=1,
        linewidth=linewidth,
        edgecolor=edgecolor,
        antialiased=True,
        shade=False,
    )

    mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(plot_data)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.1)
    cbar.set_label('log₁₀|f| (a.u.)' if log_scale else 'f (a.u.)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('log₁₀|f|' if log_scale else 'f')

    ax.set_title(f"{title}  [{slice_axis}={slice_val:.2f} a.u.]")

    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()


def plot_angle_slice(
    tree_x: vp.FunctionTree,
    tree_y: vp.FunctionTree,
    box: float,
    n: int = 80,
    slice_axis: str = 'z',
    slice_val: float = 0.0,
    cmap: str = 'RdBu_r',
    log_scale: bool = False,
    contours: bool = False,
    n_contours: int = 20,
    title: str = "Scalar field slice",
    vmin: float = None,
    vmax: float = None,
) -> None:
    """
    Same as plot_scalar_slice but plots the angle (arctan2) between two FunctionTrees
    instead of the raw scalar value. Useful for visualising the relative phase
    between two ClifFunc components, for example.
    """
    offset = 0.05 * box
    coords = np.linspace(-box + offset, box - offset, n)
    F = np.zeros((n, n))

    if slice_axis == 'z':
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                F[i, j] = np.arctan2(tree_y([a, b, slice_val]), tree_x([a, b, slice_val]))
        xlabel, ylabel = 'x (a.u.)', 'y (a.u.)'
        A, B = np.meshgrid(coords, coords, indexing='ij')
    elif slice_axis == 'y':
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                F[i, j] = np.arctan2(tree_y([a, b, slice_val]), tree_x([a, b, slice_val]))
        xlabel, ylabel = 'x (a.u.)', 'z (a.u.)'
        A, B = np.meshgrid(coords, coords, indexing='ij')
    else:  # 'x'
        for i, a in enumerate(coords):
            for j, b in enumerate(coords):
                F[i, j] =  np.arctan2(tree_y([a, b, slice_val]), tree_x([a, b, slice_val]))
        xlabel, ylabel = 'y (a.u.)', 'z (a.u.)'
        A, B = np.meshgrid(coords, coords, indexing='ij')

    plot_data = np.log10(np.abs(F) + 1e-30) if log_scale else F

    # Auto symmetric color limits for signed data
    if vmin is None and vmax is None and not log_scale:
        absmax = np.max(np.abs(plot_data))
        vmin, vmax = -absmax, absmax

    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(A, B, plot_data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('log₁₀|f| (a.u.)' if log_scale else 'f (a.u.)')

    if contours:
        levels = np.linspace(np.min(plot_data), np.max(plot_data), n_contours)
        ax.contour(A, B, plot_data, levels=levels, colors='k', linewidths=0.4, alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  [{slice_axis}={slice_val:.2f} a.u.]")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_isosurface_3d(
    tree: vp.FunctionTree,
    box: float,
    n: int = 40,
    isovals: list = None,
    colorscale: str = 'RdBu',
    log_scale: bool = False,
    opacity: float = 0.6,
    title: str = "3D isosurface",
    caps: bool = False,
) -> None:
    """
    Interactive 3-D isosurface plot of a scalar FunctionTree using Plotly.

    Samples the field on an n^3 grid and renders ``go.Isosurface``.
    Requires: ``pip install plotly``

    Parameters
    ----------
    tree : vp.FunctionTree
        The real scalar field (density, potential, spinor component, ...).
    box : float
        Half-width of the MRA box in atomic units.
    n : int
        Grid resolution per axis. Total evaluations = n^3.
        Recommended: 30–50 for density, 20–30 for potentials.
        Keep ≤ 60 for interactive performance.
    isovals : list of float, optional
        Isosurface levels to render. If None, three levels are chosen
        automatically at ±10% and ±50% of the field's max absolute value.
        Example: isovals=[0.01, 0.001, -0.001, -0.01]
    colorscale : str
        Plotly colorscale name. Use 'RdBu' for signed, 'Viridis' for density.
    log_scale : bool
        If True, plot log10(|field|) and set isovals in log units.
    opacity : float
        Surface opacity in [0, 1]. Lower values show inner surfaces.
    title : str
        Plot title.
    caps : bool
        Whether to show the flat caps on the volume boundary faces.
        Usually False for cleaner renders.

    Notes
    -----
    For a 1s hydrogen-like orbital density at Z=80, good starting isovals
    are [1e-3, 1e-2, 1e-1] (positive-definite, use colorscale='Viridis').
    For a signed function (wavefunction component), use symmetric values
    like [-0.1, -0.01, 0.01, 0.1] with colorscale='RdBu'.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Install plotly:  pip install plotly")

    xs, ys, zs, X, Y, Z = make_grid(box, n)

    print(f"Sampling {n**3} grid points... (n={n}, box={box:.1f} a.u.)")
    F = sample_tree_on_grid(tree, xs, ys, zs)

    plot_data = np.log10(np.abs(F) + 1e-30) if log_scale else F
    flat = plot_data.ravel()

    # Auto isovalues if not given
    if isovals is None:
        absmax = np.max(np.abs(flat))
        if log_scale:
            peak = np.max(flat)
            isovals = [peak - 1.0, peak - 0.5, peak - 0.1]
        else:
            isovals = sorted([
                -0.5 * absmax, -0.1 * absmax,
                 0.1 * absmax,  0.5 * absmax
            ])
        print(f"Auto isovals: {[f'{v:.3e}' for v in isovals]}")

    cap_args = dict(
        caps=dict(
            x_show=caps, y_show=caps, z_show=caps
        )
    ) if caps is not None else {}

    fig = go.Figure(data=go.Isosurface(
        x=X.ravel().astype(float),
        y=Y.ravel().astype(float),
        z=Z.ravel().astype(float),
        value=flat.astype(float),
        isomin=float(min(isovals)),
        isomax=float(max(isovals)),
        surface_count=len(isovals),
        colorscale=colorscale,
        opacity=opacity,
        colorbar=dict(title='log₁₀|f|' if log_scale else 'f (a.u.)'),
        **cap_args,
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x (a.u.)',
            yaxis_title='y (a.u.)',
            zaxis_title='z (a.u.)',
            aspectmode='cube',
        ),
    )
    fig.show()