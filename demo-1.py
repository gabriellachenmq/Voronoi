import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.optimize import least_squares

np.random.seed(0)

# ------------------------------------------------------------
# Step 1: Generate random generator points
# ------------------------------------------------------------
n = 10
true_points = np.random.rand(n, 2)

# ------------------------------------------------------------
# Step 2: Build Voronoi diagram
# ------------------------------------------------------------
vor = Voronoi(true_points)

ridge_points = vor.ridge_points
ridge_vertices = vor.ridge_vertices
vertices = vor.vertices


# ------------------------------------------------------------
# Step 3: Build constraints from Voronoi vertex-to-generator
#         equidistance (the core Voronoi property)
# ------------------------------------------------------------

def build_vertex_generator_map(vor):
    """
    For each finite Voronoi vertex, find all generator indices
    that are equidistant to it (i.e., the generators whose cells
    share that vertex).
    """
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

# Print diagnostic info
print(f"Number of generators: {n}")
print(f"Number of finite Voronoi vertices: {len(vertex_to_generators)}")
total_constraints = sum(len(gens) - 1 for gens in vertex_to_generators.values())
print(f"Number of equidistance constraints: {total_constraints}")
print(f"Number of unknowns: {2 * n}")


# ------------------------------------------------------------
# Step 4: Define residuals using equidistance from Voronoi vertices
# ------------------------------------------------------------

def residuals(flat_points):
    pts = flat_points.reshape(-1, 2)
    res = []

    # Core Voronoi property: each Voronoi vertex is equidistant
    # from all generators whose cells meet at that vertex.
    # For each vertex v with generators {g0, g1, g2, ...}:
    #   ||g1 - v||^2 - ||g0 - v||^2 = 0
    #   ||g2 - v||^2 - ||g0 - v||^2 = 0
    #   ...
    for vi, gens in vertex_to_generators.items():
        v = vertices[vi]
        gen_list = sorted(gens)
        g0 = pts[gen_list[0]]
        d0 = np.sum((g0 - v) ** 2)

        for gidx in gen_list[1:]:
            gi = pts[gidx]
            di = np.sum((gi - v) ** 2)
            res.append(di - d0)

    # Also use edge perpendicularity for finite ridges
    # (generator difference is perpendicular to Voronoi edge)
    for (i, j), rv in zip(ridge_points, ridge_vertices):
        if -1 in rv:
            continue
        v1, v2 = vertices[rv[0]], vertices[rv[1]]
        edge_dir = v2 - v1
        pi = pts[i]
        pj = pts[j]
        res.append(np.dot(pi - pj, edge_dir))

    # Also: midpoint of (pi, pj) lies on the Voronoi edge line
    # This means the midpoint of two neighboring generators lies
    # on the line through the Voronoi edge.
    for (i, j), rv in zip(ridge_points, ridge_vertices):
        if -1 in rv:
            continue
        v1, v2 = vertices[rv[0]], vertices[rv[1]]
        edge_dir = v2 - v1
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-15:
            continue
        edge_normal = np.array([-edge_dir[1], edge_dir[0]]) / edge_len

        pi = pts[i]
        pj = pts[j]
        mid = 0.5 * (pi + pj)

        # The midpoint should lie on the line through v1, v2
        # i.e., (mid - v1) has zero component along edge_normal
        res.append(np.dot(mid - v1, edge_normal))

    return np.array(res)


# ------------------------------------------------------------
# Step 5: Reconstruct generators
# ------------------------------------------------------------

# Start from a perturbation of the true points (simulating "forgotten" generators)
# In a real scenario you'd use a smarter initialization
initial_guess = true_points.flatten() + 0.05 * np.random.randn(n * 2)

result = least_squares(
    residuals,
    initial_guess,
    method='lm',  # Levenberg-Marquardt is good for nonlinear least squares
    ftol=1e-15,
    xtol=1e-15,
    gtol=1e-15,
    max_nfev=10000
)

reconstructed = result.x.reshape(-1, 2)

print(f"\nOptimizer cost: {result.cost:.2e}")
print(f"Optimizer status: {result.message}")


# ------------------------------------------------------------
# Step 6: Align reconstruction to original via Procrustes
# (handles any residual translation/rotation/reflection)
# ------------------------------------------------------------

def procrustes_align(X, Y):
    """Align X to Y using Procrustes (translation + rotation, no scaling)."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    U, S, Vt = np.linalg.svd(Xc.T @ Yc)

    # Handle reflection
    d = np.linalg.det(U @ Vt)
    D = np.diag([1, np.sign(d)])
    R = U @ D @ Vt

    X_aligned = Xc @ R + Y.mean(axis=0)
    return X_aligned


reconstructed_aligned = procrustes_align(reconstructed, true_points)

# ------------------------------------------------------------
# Step 7: Compute reconstruction error
# ------------------------------------------------------------

# Per-point errors
per_point_error = np.linalg.norm(true_points - reconstructed_aligned, axis=1)
rms_error = np.sqrt(np.mean(per_point_error ** 2))

print(f"\nPer-point errors: {per_point_error}")
print(f"RMS reconstruction error: {rms_error:.6e}")
print(f"Max reconstruction error: {np.max(per_point_error):.6e}")

# ------------------------------------------------------------
# Step 8: Visualization
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Voronoi diagram with original and reconstructed
ax1 = axes[0]
from scipy.spatial import voronoi_plot_2d

voronoi_plot_2d(vor, ax=ax1, show_vertices=True, line_colors='lightgray',
                line_width=0.8, point_size=0)
ax1.scatter(true_points[:, 0], true_points[:, 1],
            c='blue', s=80, zorder=5, label="Original")
ax1.scatter(reconstructed_aligned[:, 0], reconstructed_aligned[:, 1],
            marker='x', c='red', s=80, zorder=5, label="Reconstructed")

for i in range(n):
    ax1.plot([true_points[i, 0], reconstructed_aligned[i, 0]],
             [true_points[i, 1], reconstructed_aligned[i, 1]],
             'k--', alpha=0.4)
    ax1.annotate(str(i), true_points[i], fontsize=8, ha='center', va='bottom')

ax1.legend()
ax1.set_title(f"Inverse Voronoi Reconstruction\nRMS Error: {rms_error:.2e}")
ax1.set_aspect('equal')
ax1.set_xlim(-0.2, 1.2)
ax1.set_ylim(-0.2, 1.2)

# Right: Error bar chart
ax2 = axes[1]
ax2.bar(range(n), per_point_error)
ax2.set_xlabel("Generator Index")
ax2.set_ylabel("Reconstruction Error")
ax2.set_title("Per-Point Reconstruction Error")

plt.tight_layout()
plt.show()