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
# Step 3: Define reconstruction residual system
# ------------------------------------------------------------

def residuals(flat_points):
    pts = flat_points.reshape(-1, 2)
    res = []

    for (i, j), rv in zip(ridge_points, ridge_vertices):

        if -1 in rv:
            continue

        v1, v2 = vertices[rv]
        edge_dir = v2 - v1
        midpoint = 0.5 * (v1 + v2)

        pi = pts[i]
        pj = pts[j]

        # Perpendicular constraint
        res.append(np.dot(pi - pj, edge_dir))

        # Equidistant constraint
        res.append(np.linalg.norm(pi - midpoint)**2 -
                   np.linalg.norm(pj - midpoint)**2)

    # --- REMOVE geometric freedoms ---

    # Fix translation
    res.append(pts[0, 0])
    res.append(pts[0, 1])

    # Fix rotation (force second point to lie on x-axis)
    res.append(pts[1, 1])

    # Fix scale (distance between p0 and p1 = 1)
    res.append(np.linalg.norm(pts[1] - pts[0]) - 1.0)

    return np.array(res)



# ------------------------------------------------------------
# Step 4: Reconstruct generators
# ------------------------------------------------------------

initial_guess = true_points.flatten() + 0.1*np.random.randn(n*2)

result = least_squares(residuals, initial_guess)

reconstructed = result.x.reshape(-1, 2)

# ------------------------------------------------------------
# Step 5: Align reconstruction to original (remove rotation/scale)
# ------------------------------------------------------------

def procrustes_align(X, Y):
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = U @ Vt

    X_aligned = Xc @ R
    return X_aligned + Y.mean(axis=0)

reconstructed_aligned = procrustes_align(reconstructed, true_points)

# ------------------------------------------------------------
# Step 6: Compute reconstruction error
# ------------------------------------------------------------

error = np.linalg.norm(true_points - reconstructed_aligned) / np.sqrt(n)

print("RMS reconstruction error:", error)

# ------------------------------------------------------------
# Step 7: Visualization
# ------------------------------------------------------------

plt.figure(figsize=(6, 6))
plt.scatter(true_points[:, 0], true_points[:, 1], label="Original", s=80)
plt.scatter(reconstructed_aligned[:, 0],
            reconstructed_aligned[:, 1],
            marker='x', s=80, label="Reconstructed")

for i in range(n):
    plt.plot([true_points[i, 0], reconstructed_aligned[i, 0]],
             [true_points[i, 1], reconstructed_aligned[i, 1]],
             'k--', alpha=0.4)

plt.legend()
plt.title("Inverse Voronoi Reconstruction")
plt.axis("equal")
plt.show()
