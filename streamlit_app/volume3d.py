"""
volume3d.py
-----------
3D reconstruction for NeuroScan's volumetric mode.

Turns a stack of per-slice 2D masks into smooth, closed, lightweight
surface meshes and assembles an interactive Plotly scene.

Pipeline
1. Every slice is put on one shared in-plane grid (tumor and brain masks
   were previously on different grids, which misaligned the two meshes).
2. The brain envelope is a *filled*, cleaned head mask rather than a thin
   outline, so it meshes into one solid shell instead of a jagged tube.
3. Spurious tumor specks are removed in 3D (connected components) and the
   tumor is constrained to lie inside the brain.
4. Slices are converted to signed distance fields and interpolated along Z
   (shape-based interpolation). Shapes morph smoothly between slices
   instead of stair-stepping, and the volume is padded so meshes close.
5. Marching cubes extracts the surface, then Taubin smoothing removes
   voxel ripples without shrinking the mesh. Face counts are budgeted so
   the browser stays responsive.

Everything here is pure NumPy/SciPy/scikit-image, independent of Streamlit
and the models, so it can be tested on synthetic data.
"""

from dataclasses import dataclass, field
import time

import cv2
import numpy as np
import plotly.graph_objects as go
from scipy import ndimage, sparse
from skimage.measure import marching_cubes


# Quality presets: in-plane working grid and per-mesh face budgets.
QUALITY_PRESETS = {
    "Fast": {"grid": 112, "brain_faces": 40_000, "tumor_faces": 30_000},
    "Balanced": {"grid": 160, "brain_faces": 90_000, "tumor_faces": 60_000},
    "High": {"grid": 224, "brain_faces": 160_000, "tumor_faces": 120_000},
}

# Upper bound on voxels in an interpolated volume (memory and time guard).
MAX_VOXELS = 14_000_000

TUMOR_COLOR = "#E0242E"


# 2D SLICE PREPROCESSING

def brain_mask_from_slice(gray):
    """
    Return a filled, cleaned binary head/brain mask for one grayscale slice.

    Otsu thresholding with a floor that ignores the dark background, then
    closing, hole filling, and keeping the largest component.
    """
    gray = np.asarray(gray, dtype=np.uint8)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(12.0, 0.45 * float(otsu))
    mask = (blur > threshold).astype(np.uint8)

    if not mask.any():
        return mask.astype(bool)

    size = max(3, int(round(min(gray.shape) * 0.025)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == largest
    else:
        mask = mask.astype(bool)

    return ndimage.binary_fill_holes(mask)


def prepare_slice(image_rgb, tumor_prob, size=224):
    """
    Bring one analyzed slice onto the shared grid.

    Returns (gray uint8, brain mask bool, tumor mask bool), all size x size.
    `tumor_prob` is the segmentation sigmoid output at any resolution.
    """
    gray = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    prob = cv2.resize(
        np.asarray(tumor_prob, dtype=np.float32), (size, size),
        interpolation=cv2.INTER_LINEAR,
    )

    return gray, brain_mask_from_slice(gray), prob > 0.5


def _resize_stack(stack, grid):
    """Resize a (n, h, w) bool stack to (n, grid, grid)."""
    if stack.shape[1] == grid and stack.shape[2] == grid:
        return stack.astype(bool)
    return np.stack([
        cv2.resize(s.astype(np.float32), (grid, grid), interpolation=cv2.INTER_LINEAR) > 0.5
        for s in stack
    ])


# 3D CLEANUP

def clean_tumor_volume(tumor, brain, min_fraction=0.04):
    """
    Remove 3D tumor specks and anything outside the (dilated) brain.

    Components smaller than `min_fraction` of the largest component, or
    below a small absolute size, are treated as segmentation noise.
    Returns (cleaned mask, number of components kept).
    """
    if not tumor.any():
        return tumor, 0

    if brain.any():
        tumor = tumor & ndimage.binary_dilation(
            brain, structure=np.ones((1, 5, 5), bool)
        )

    labels, count = ndimage.label(tumor, structure=np.ones((3, 3, 3), bool))
    if count == 0:
        return tumor, 0

    sizes = np.bincount(labels.ravel())[1:]
    min_size = max(sizes.max() * min_fraction, tumor.shape[1] * tumor.shape[2] * 0.0004)
    keep = np.flatnonzero(sizes >= min_size) + 1

    return np.isin(labels, keep), int(len(keep))


# SHAPE-BASED INTERPOLATION

def _signed_distance(mask, cap):
    """2D signed distance: negative inside, positive outside, clipped."""
    if not mask.any():
        return np.full(mask.shape, cap, np.float32)
    if mask.all():
        return np.full(mask.shape, -cap, np.float32)
    sdf = ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)
    return np.clip(sdf, -cap, cap).astype(np.float32)


def interpolated_field(stack, factor, pad_xy=2, cap=24.0, sigma=1.0):
    """
    Build an "inside-positive" scalar field from a (n, h, w) mask stack.

    Each slice becomes a signed distance field. `factor - 1` intermediate
    layers are blended linearly between neighbors, which morphs shapes
    smoothly across the gap. The volume is padded with "outside" so the
    isosurface closes at the first and last slice.

    Returns (field, pad_z, pad_xy). Original slice k is at Z index
    pad_z + k * factor.
    """
    n = stack.shape[0]
    sdfs = np.stack([_signed_distance(s, cap) for s in stack])

    if n > 1 and factor > 1:
        t = (np.arange(factor, dtype=np.float32) / factor)[None, :, None, None]
        blended = (1 - t) * sdfs[:-1, None] + t * sdfs[1:, None]
        sdfs = np.concatenate(
            [blended.reshape(-1, *sdfs.shape[1:]), sdfs[-1:]], axis=0
        )

    pad_z = max(2, factor)
    field = np.pad(
        sdfs, ((pad_z, pad_z), (pad_xy, pad_xy), (pad_xy, pad_xy)),
        constant_values=cap,
    )

    if sigma > 0:
        field = ndimage.gaussian_filter(field, sigma=sigma, mode="nearest")

    return -field, pad_z, pad_xy


# MESH EXTRACTION AND SMOOTHING

def _neighbor_operator(faces, n_verts):
    """Row-normalized vertex adjacency matrix (uniform Laplacian weights)."""
    a = faces[:, [0, 1, 2, 1, 2, 0]].ravel()
    b = faces[:, [1, 2, 0, 0, 1, 2]].ravel()
    adj = sparse.coo_matrix(
        (np.ones(a.size, np.float32), (a, b)), shape=(n_verts, n_verts)
    ).tocsr()
    adj.data[:] = 1.0
    degree = np.asarray(adj.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    return sparse.diags(1.0 / degree) @ adj


def taubin_smooth(verts, faces, iterations=12, lam=0.5, mu=-0.53):
    """
    Taubin smoothing: alternating shrink/inflate Laplacian steps that remove
    high-frequency ripples (voxel artifacts) without shrinking the volume.
    """
    if iterations <= 0 or len(faces) == 0:
        return verts
    op = _neighbor_operator(faces, len(verts))
    v = verts.astype(np.float64)
    for _ in range(iterations):
        v = v + lam * (op @ v - v)
        v = v + mu * (op @ v - v)
    return v.astype(np.float32)


@dataclass
class Mesh:
    verts: np.ndarray            # (n, 3) as x, y, z in display coordinates
    faces: np.ndarray            # (m, 3) int32
    step_size: int = 1
    stats: dict = field(default_factory=dict)

    @property
    def n_faces(self):
        return int(len(self.faces))


def extract_mesh(field, spacing, pad, height, face_budget, smooth_iters=12, step_size=1):
    """
    Marching cubes at level 0 with an adaptive step size to respect the
    face budget, then Taubin smoothing and conversion to display coords.

    `spacing` is (dz, dy, dx). `pad` is (pad_z, pad_xy). `height` is the
    in-plane grid size, used to flip rows so images stand upright.
    Returns None if no surface exists.
    """
    if field.max() <= 0 or field.min() >= 0:
        return None

    step = max(1, int(step_size))
    while True:
        verts, faces, _, _ = marching_cubes(
            field, level=0.0, spacing=spacing,
            step_size=step, allow_degenerate=False,
        )
        if len(faces) <= face_budget or step >= 4:
            break
        # Face count scales roughly with 1 / step^2.
        step = max(step + 1, int(np.ceil(step * np.sqrt(len(faces) / face_budget))))
        step = min(step, 4)

    faces = faces.astype(np.int32)
    verts = taubin_smooth(verts, faces, iterations=smooth_iters)

    pad_z, pad_xy = pad
    dz, dy, dx = spacing
    z = verts[:, 0] - pad_z * dz
    row = verts[:, 1] - pad_xy * dy
    col = verts[:, 2] - pad_xy * dx

    display = np.column_stack([col, (height - 1) - row, z]).astype(np.float32)
    return Mesh(verts=display, faces=faces, step_size=step)


def mesh_volume(mesh):
    """Enclosed volume of a closed triangle mesh (divergence theorem)."""
    v = mesh.verts.astype(np.float64)
    a, b, c = v[mesh.faces[:, 0]], v[mesh.faces[:, 1]], v[mesh.faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


# FULL RECONSTRUCTION

def reconstruct(brain_stack, tumor_stack, slice_spacing=3.0, quality="Balanced", smoothness=12):
    """
    Build the brain envelope and tumor meshes.

    brain_stack, tumor_stack: (n, h, w) bool arrays on the same grid.
    slice_spacing: distance between slices, in units of the 224-px grid.

    Returns a dict with "brain" / "tumor" Mesh objects (or None), the grid
    size, spacing and summary statistics.
    """
    start = time.perf_counter()
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Balanced"])
    grid = preset["grid"]

    brain = _resize_stack(np.asarray(brain_stack, bool), grid)
    tumor = _resize_stack(np.asarray(tumor_stack, bool), grid)
    tumor, components = clean_tumor_volume(tumor, brain)

    n = brain.shape[0]
    # Slice spacing is given relative to the 224 grid; rescale to this grid.
    dz_grid = float(slice_spacing) * grid / 224.0

    # Interpolate to roughly isotropic voxels, within a voxel budget.
    factor = int(np.clip(round(dz_grid), 1, 8)) if n > 1 else 1
    while factor > 1 and (n * factor) * (grid + 4) ** 2 > MAX_VOXELS:
        factor -= 1
    spacing = (dz_grid / factor, 1.0, 1.0)

    result = {
        "grid": grid,
        "n_slices": n,
        "slice_spacing": dz_grid,
        "components": components,
        "brain": None,
        "tumor": None,
    }

    if brain.any():
        field, pad_z, pad_xy = interpolated_field(brain, factor, cap=24.0, sigma=1.5)
        result["brain"] = extract_mesh(
            field, spacing, (pad_z, pad_xy), grid,
            face_budget=preset["brain_faces"], smooth_iters=smoothness,
        )
        del field

    if tumor.any():
        # Crop to the tumor's bounding box so small tumors are cheap to mesh.
        zs, ys, xs = np.nonzero(tumor)
        m = 3
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, grid)
        x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, grid)
        z0, z1 = zs.min(), zs.max() + 1
        crop = tumor[z0:z1, y0:y1, x0:x1]

        field, pad_z, pad_xy = interpolated_field(crop, factor, cap=12.0, sigma=1.0)
        mesh = extract_mesh(
            field, spacing, (pad_z, pad_xy), grid,
            face_budget=preset["tumor_faces"], smooth_iters=smoothness,
        )
        if mesh is not None:
            # Shift from crop coordinates back into the full grid.
            mesh.verts[:, 0] += x0
            mesh.verts[:, 1] -= y0
            mesh.verts[:, 2] += z0 * dz_grid
            # Report volume in 224-grid pixel units, independent of quality.
            mesh.stats["volume_px3"] = mesh_volume(mesh) * (224.0 / grid) ** 3
            mesh.stats["slice_range"] = (int(z0), int(z1 - 1))
        result["tumor"] = mesh

    result["seconds"] = time.perf_counter() - start
    return result


# SLICE SPACING ESTIMATE

DEFAULT_SLICE_SPACING = 3.0


def estimate_slice_spacing(brain_stack, depth_ratio=0.8):
    """
    Estimate slice spacing (in 224-grid pixels) from how the head's
    cross-section changes through the sequence.

    The head is modeled as an ellipsoid whose half-depth is `depth_ratio`
    times its in-plane radius. For an ellipsoid, the squared cross-section
    radius is a downward parabola in slice position, r_k^2 = R^2 - (R s / c)^2
    (k - k0)^2, so fitting a quadratic to r_k^2 recovers the spacing s.
    This works for partial sequences too, as long as the head visibly
    narrows somewhere. Returns None when there isn't enough signal.
    """
    stack = np.asarray(brain_stack, bool)
    n = len(stack)
    if n < 4:
        return None

    scale = 224.0 / stack.shape[1]
    area = stack.reshape(n, -1).sum(axis=1) * scale ** 2
    valid = area > 0.02 * area.max()
    if valid.sum() < 4:
        return None

    k = np.flatnonzero(valid).astype(np.float64)
    r2 = area[valid] / np.pi
    q, _, _ = np.polyfit(k, r2, 2)
    if q >= 0:
        return None

    # -q = R^2 s^2 / c^2 with c = depth_ratio * R  =>  s = depth_ratio * sqrt(-q)
    spacing = depth_ratio * np.sqrt(-q)
    return float(np.clip(spacing, 1.0, 12.0))


# PLOTLY SCENE

def _brain_outline(mask, smooth=7, step=2.0):
    """Largest contour of a mask, smoothed and resampled every `step` px."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    if len(c) < 8:
        return None
    # Circular moving average removes the pixel staircase.
    kernel = np.ones(smooth) / smooth
    c = np.column_stack([
        np.convolve(np.r_[c[-smooth:, i], c[:, i], c[:smooth, i]], kernel, "same")[smooth:-smooth]
        for i in range(2)
    ])
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    samples = np.arange(0, arc[-1], step)
    closed = np.vstack([c, c[:1]])
    return np.column_stack([np.interp(samples, arc, closed[:, i]) for i in range(2)])


def _slice_plane(gray, brain, index, z, grid, max_res=160):
    """
    Grayscale MRI slice as a flat mesh clipped to a smooth brain outline.

    A Delaunay triangulation of interior pixels plus a smoothed contour
    gives a clean edge (a masked go.Surface leaves a jagged sawtooth).
    """
    from scipy.spatial import Delaunay

    res = min(grid, max_res)
    img = cv2.resize(gray[index], (res, res), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = cv2.resize(brain[index].astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST) > 0
    if not mask.any():
        return None

    # Trace the outline slightly inside the edge so the rim never samples
    # the dark background just outside the skull.
    outline = _brain_outline(ndimage.binary_erosion(mask, iterations=1))
    if outline is None:
        return None

    inner_mask = ndimage.binary_erosion(mask, iterations=3)
    iy, ix = np.nonzero(inner_mask)
    pts = np.vstack([outline, np.column_stack([ix, iy]).astype(np.float64)])

    tri = Delaunay(pts).simplices
    # Drop triangles whose centroid falls outside the outline (concavities).
    path_mask = np.zeros_like(mask, np.uint8)
    cv2.fillPoly(path_mask, [np.round(outline).astype(np.int32)], 1)
    cen = pts[tri].mean(axis=1)
    cx = np.clip(np.round(cen[:, 0]).astype(int), 0, res - 1)
    cy = np.clip(np.round(cen[:, 1]).astype(int), 0, res - 1)
    tri = tri[path_mask[cy, cx] > 0]

    values = ndimage.map_coordinates(img, [pts[:, 1], pts[:, 0]], order=1, mode="nearest")
    lo, hi = np.percentile(img[mask], [1, 99.5])

    scale = (grid - 1) / (res - 1)
    return go.Mesh3d(
        x=pts[:, 0] * scale,
        y=(grid - 1) - pts[:, 1] * scale,
        z=np.full(len(pts), z, np.float32),
        i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
        intensity=values,
        colorscale="gray",
        cmin=lo, cmax=max(hi, lo + 1),
        showscale=False,
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0),
        name=f"MRI slice {index + 1}",
        showlegend=True,
        hoverinfo="skip",
    )


def build_figure(recon, gray_stack=None, brain_stack=None, show_brain=True, slice_index=None):
    """Assemble the interactive scene from a `reconstruct` result."""
    fig = go.Figure()
    grid = recon["grid"]

    brain = recon["brain"]
    if show_brain and brain is not None:
        v, f = brain.verts, brain.faces
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color="#A3B1C6",
            opacity=0.17,
            flatshading=False,
            lighting=dict(ambient=0.5, diffuse=0.6, specular=0.3,
                          roughness=0.5, fresnel=0.5),
            lightposition=dict(x=200, y=300, z=400),
            name="Brain envelope",
            showlegend=True,
            hoverinfo="skip",
        ))

    if slice_index is not None and gray_stack is not None and brain_stack is not None:
        plane = _slice_plane(
            gray_stack, brain_stack, slice_index,
            slice_index * recon["slice_spacing"], grid,
        )
        if plane is not None:
            fig.add_trace(plane)

    tumor = recon["tumor"]
    if tumor is not None:
        v, f = tumor.verts, tumor.faces
        # Solid color: the lighting alone carries the shape. (A position-based
        # color ramp made flat masses read as a ring instead of a solid.)
        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=TUMOR_COLOR,
            opacity=1.0,
            flatshading=False,
            lighting=dict(ambient=0.3, diffuse=0.9, specular=0.4,
                          roughness=0.4, fresnel=0.2),
            lightposition=dict(x=1000, y=-1000, z=2000),
            name="Tumor mass",
            showlegend=True,
            hovertemplate="Tumor mass<extra></extra>",
        ))

    return fig


def best_slice_index(tumor_stack, brain_stack):
    """Slice with the largest tumor area, else the largest brain cross-section."""
    tumor_area = np.asarray(tumor_stack).reshape(len(tumor_stack), -1).sum(axis=1)
    if tumor_area.max() > 0:
        return int(np.argmax(tumor_area))
    brain_area = np.asarray(brain_stack).reshape(len(brain_stack), -1).sum(axis=1)
    return int(np.argmax(brain_area))
