import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.optimize import least_squares

"""

ALGORITHM: Reconstruct Voronoi Generators from Diagram
════════════════════════════════════════════════════════

INPUT:  Voronoi diagram (vertices, edges, cell topology)
OUTPUT: Reconstructed generator points

────────────────────────────────────────────────────────

1. INITIAL GUESS
   For each cell:
       Estimate generator as centroid of cell's vertices

2. DEFINE CONSTRAINTS
   For each Voronoi vertex:
       All neighboring generators must be equidistant from it
   For each Voronoi edge:
       Its two neighboring generators must be perpendicular to it
       The midpoint of its two generators must lie on it

3. OPTIMIZE
   Adjust generator positions to satisfy all constraints
   using least-squares optimization (Levenberg-Marquardt)

4. EVALUATE
   Align reconstructed generators to true generators
   Compute RMS error

"""

np.random.seed(0)
n = 10
true_points = np.random.rand(n, 2)

vor = Voronoi(true_points)

ridge_points = vor.ridge_points      # cell adjacency
ridge_vertices = vor.ridge_vertices  # edge-to-vertex mapping
vertices = vor.vertices              # vertex coordinates


def build_vertex_generator_map(vor):
    vertex_to_generators = {}
    for (i, j), rv in zip(vor.ridge_points, vor.ridge_vertices):
        for vi in rv:
            if vi == -1:
                continue
            if vi not in vertex_to_generators:
                vertex_to_generators[vi] = set()
            vertex_to_generators[vi].add(i)
            vertex_to_generators[vi].add(j)
    return vertex_to_generators

vertex_to_generators = build_vertex_generator_map(vor)


def residuals(flat_points):
    pts = flat_points.reshape(-1, 2)
    res = []

    # Constraint 1: Vertex equidistance
    for vi, gens in vertex_to_generators.items():
        v = vertices[vi]
        gen_list = sorted(gens)
        g0 = pts[gen_list[0]]
        d0 = np.sum((g0 - v) ** 2)
        for gidx in gen_list[1:]:
            gi = pts[gidx]
            di = np.sum((gi - v) ** 2)
            res.append(di - d0)

    # Constraint 2: Edge perpendicularity
    for (i, j), rv in zip(ridge_points, ridge_vertices):
        if -1 in rv:
            continue
        v1, v2 = vertices[rv[0]], vertices[rv[1]]
        edge_dir = v2 - v1
        res.append(np.dot(pts[i] - pts[j], edge_dir))

    # Constraint 3: Generator midpoint lies on Voronoi edge line
    for (i, j), rv in zip(ridge_points, ridge_vertices):
        if -1 in rv:
            continue
        v1, v2 = vertices[rv[0]], vertices[rv[1]]
        edge_dir = v2 - v1
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-15:
            continue
        edge_normal = np.array([-edge_dir[1], edge_dir[0]]) / edge_len
        mid = 0.5 * (pts[i] + pts[j])
        res.append(np.dot(mid - v1, edge_normal))

    return np.array(res)


def initialize_from_voronoi(vor):
    n = len(vor.point_region)
    initial = np.zeros((n, 2))
    for gen_idx in range(n):
        region_idx = vor.point_region[gen_idx]
        region = vor.regions[region_idx]
        finite_verts = [v for v in region if v != -1]
        if len(finite_verts) == 0:
            initial[gen_idx] = vor.vertices.mean(axis=0)
        else:
            initial[gen_idx] = vor.vertices[finite_verts].mean(axis=0)
    return initial

initial_guess = initialize_from_voronoi(vor).flatten()

# Show how far the initial guess is from truth
init_shaped = initial_guess.reshape(-1, 2)
print("Initial guess distances from true points:")
for i in range(n):
    d = np.linalg.norm(init_shaped[i] - true_points[i])
    print(f"  Generator {i}: {d:.4f}")


result = least_squares(
    residuals,
    initial_guess,
    method='lm',
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
    max_nfev=10000
)

reconstructed = result.x.reshape(-1, 2)
print(f"\nOptimizer cost: {result.cost:.2e}")
print(f"Optimizer status: {result.message}")


def procrustes_align(X, Y):
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1, np.sign(d)])
    R = U @ D @ Vt
    return Xc @ R + Y.mean(axis=0)

reconstructed_aligned = procrustes_align(reconstructed, true_points)


per_point_error = np.linalg.norm(
    true_points - reconstructed_aligned, axis=1
)
rms_error = np.sqrt(np.mean(per_point_error ** 2))

print(f"\nRMS reconstruction error: {rms_error:.6e}")
print(f"Max reconstruction error: {np.max(per_point_error):.6e}")


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: Initial guess vs truth
ax0 = axes[0]
voronoi_plot_2d(vor, ax=ax0, show_vertices=True,
                line_colors='lightgray', line_width=0.8, point_size=0)
ax0.scatter(*true_points.T, c='blue', s=80, label="True", zorder=5)
ax0.scatter(*init_shaped.T, marker='^', c='green',
            s=80, label="Initial Guess", zorder=5)
ax0.legend()
ax0.set_title("Initial Guess (cell centroids)")
ax0.set_aspect('equal')
ax0.set_xlim(-0.2, 1.2)
ax0.set_ylim(-0.2, 1.2)

# Middle: Reconstructed vs truth
ax1 = axes[1]
voronoi_plot_2d(vor, ax=ax1, show_vertices=True,
                line_colors='lightgray', line_width=0.8, point_size=0)
ax1.scatter(*true_points.T, c='blue', s=80, label="True", zorder=5)
ax1.scatter(*reconstructed_aligned.T, marker='x', c='red',
            s=80, label="Reconstructed", zorder=5)
for i in range(n):
    ax1.plot([true_points[i, 0], reconstructed_aligned[i, 0]],
             [true_points[i, 1], reconstructed_aligned[i, 1]],
             'k--', alpha=0.4)
ax1.legend()
ax1.set_title(f"Reconstruction\nRMS Error: {rms_error:.2e}")
ax1.set_aspect('equal')
ax1.set_xlim(-0.2, 1.2)
ax1.set_ylim(-0.2, 1.2)

# Right: Error bar chart
ax2 = axes[2]
ax2.bar(range(n), per_point_error)
ax2.set_xlabel("Generator Index")
ax2.set_ylabel("Error")
ax2.set_title("Per-Point Error")

plt.tight_layout()
plt.show()