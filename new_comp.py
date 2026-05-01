import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

# =============================================================================
# TORUS UTILITIES
# =============================================================================

def tile_points(pts, Lx, Ly):
    shifts = np.array([
        [dx, dy]
        for dx in [-Lx, 0, Lx]
        for dy in [-Ly, 0, Ly]
    ])
    copies = [pts + s for s in shifts]
    return np.vstack(copies)

def voronoi_on_torus(pts, Lx, Ly):
    tiled = tile_points(pts, Lx, Ly)
    return Voronoi(tiled)

def get_original_index_offset(N):
    return 4 * N

# =============================================================================
# GEOMETRY
# =============================================================================

def polygon_centroid(vertices):
    """Area centroid (Lloyd)"""
    x = vertices[:, 0]
    y = vertices[:, 1]

    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    A = 0.5 * a

    if np.abs(A) < 1e-12:
        return vertices.mean(axis=0)

    cx = np.sum((x + np.roll(x, -1)) *
                (x * np.roll(y, -1) - np.roll(x, -1) * y))
    cy = np.sum((y + np.roll(y, -1)) *
                (x * np.roll(y, -1) - np.roll(x, -1) * y))

    cx /= (6 * A)
    cy /= (6 * A)

    return np.array([cx, cy])

# =============================================================================
# ENERGY
# =============================================================================

def voronoi_energy(X, vor, offset):
    """Simple discrete energy (good enough for comparison)"""
    E = 0.0
    m = len(X)

    for i in range(m):
        global_idx = offset + i
        region_idx = vor.point_region[global_idx]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            continue

        V = vor.vertices[region]
        xi = X[i]

        E += np.sum(np.linalg.norm(V - xi, axis=1)**2)

    return E

# =============================================================================
# CORE ITERATION ENGINE
# =============================================================================

def run_voronoi_method(X_init, Lx, Ly, method="divi",
                       eps=1e-5, max_iter=300):
    """
    method = "divi"  → vertex average
    method = "lloyd" → polygon centroid
    """

    X = X_init.copy()
    m = len(X)

    history = [X.copy()]
    errors = []
    energies = []

    offset = get_original_index_offset(m)

    for it in range(max_iter):

        vor = voronoi_on_torus(X, Lx, Ly)

        # --- Energy ---
        energies.append(voronoi_energy(X, vor, offset))

        X_new = X.copy()

        for i in range(m):
            global_idx = offset + i
            region_idx = vor.point_region[global_idx]
            region = vor.regions[region_idx]

            if -1 in region or len(region) == 0:
                continue

            V = vor.vertices[region]
            if len(V) < 3:
                continue

            if method == "divi":
                c = V.mean(axis=0)
            elif method == "lloyd":
                c = polygon_centroid(V)
            else:
                raise ValueError("Unknown method")

            X_new[i] = [c[0] % Lx, c[1] % Ly]

        # --- Toroidal distance ---
        diff = X_new - X
        diff[:, 0] -= Lx * np.round(diff[:, 0] / Lx)
        diff[:, 1] -= Ly * np.round(diff[:, 1] / Ly)

        max_move = np.max(np.linalg.norm(diff, axis=1))

        errors.append(max_move)
        X = X_new
        history.append(X.copy())

        print(f"{method:6s} | iter {it:3d} | move = {max_move:.6e}")

        if max_move < eps:
            print(f"{method} converged at iteration {it}")
            break

    return X, history, errors, energies

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_torus_voronoi(X, Lx, Ly, ax, title=''):
    vor = voronoi_on_torus(X, Lx, Ly)

    margin = 0.01
    for simplex in vor.ridge_vertices:
        if -1 in simplex:
            continue
        v0, v1 = vor.vertices[simplex]
        if (-margin <= v0[0] <= Lx+margin and -margin <= v0[1] <= Ly+margin and
            -margin <= v1[0] <= Lx+margin and -margin <= v1[1] <= Ly+margin):
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], lw=0.5, alpha=0.5)

    ax.scatter(X[:, 0], X[:, 1], s=20, edgecolors='k')

    rect = plt.Rectangle((0, 0), Lx, Ly,
                         fill=False, edgecolor='black', lw=1.5, ls='--')
    ax.add_patch(rect)

    ax.set(xlim=(-0.3, Lx+0.3),
           ylim=(-0.3, Ly+0.3),
           aspect='equal',
           title=title)

# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':

    Lx, Ly = 10.0, 10.0
    M = 100

    # SAME INITIAL CONDITION
    np.random.seed(42)
    X0 = np.c_[
        np.random.uniform(0, Lx, M),
        np.random.uniform(0, Ly, M),
    ]

    print("\nRunning DIVI...")
    X_divi, hist_divi, err_divi, E_divi = run_voronoi_method(
        X0, Lx, Ly, method="divi"
    )

    print("\nRunning Lloyd...")
    X_lloyd, hist_lloyd, err_lloyd, E_lloyd = run_voronoi_method(
        X0, Lx, Ly, method="lloyd"
    )

    # =============================================================================
    # SIDE-BY-SIDE COMPARISON
    # =============================================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_torus_voronoi(hist_divi[-1], Lx, Ly, axes[0], title='DIVI')
    plot_torus_voronoi(hist_lloyd[-1], Lx, Ly, axes[1], title='Lloyd')

    plt.tight_layout()
    plt.show()

    # =============================================================================
    # OVERLAY COMPARISON
    # =============================================================================

    plt.figure(figsize=(6, 6))
    plt.scatter(hist_divi[-1][:, 0], hist_divi[-1][:, 1],
                label='DIVI', s=25)
    plt.scatter(hist_lloyd[-1][:, 0], hist_lloyd[-1][:, 1],
                label='Lloyd', s=25)

    plt.legend()
    plt.title("Final Generator Positions")
    plt.xlim(0, Lx)
    plt.ylim(0, Ly)
    plt.gca().set_aspect('equal')
    plt.show()

    # =============================================================================
    # ENERGY COMPARISON
    # =============================================================================

    plt.figure(figsize=(7, 4))
    plt.plot(E_divi, label='DIVI')
    plt.plot(E_lloyd, label='Lloyd')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Voronoi Energy Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()