import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi


def tile_points(pts, Lx, Ly):
    shifts = np.array([[dx, dy] for dx in [-Lx, 0, Lx] for dy in [-Ly, 0, Ly]])
    return np.vstack([pts + s for s in shifts])

def voronoi_on_torus(pts, Lx, Ly):
    return Voronoi(tile_points(pts, Lx, Ly))

def get_original_index_offset(N):
    """The (0,0) copy sits at index 4 in our 3×3 tiling."""
    return 4 * N



def polygon_area_centroid(V):

    x, y = V[:, 0], V[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = 0.5 * cross.sum()
    if abs(A) < 1e-14:
        return 0.0, V.mean(axis=0)
    Cx = ((x + x1) * cross).sum() / (6.0 * A)
    Cy = ((y + y1) * cross).sum() / (6.0 * A)
    return A, np.array([Cx, Cy])

def polygon_second_moment(V, p):

    energy = 0.0
    n = len(V)
    for j in range(n):
        a = V[j] - p
        b = V[(j + 1) % n] - p
        tri_A = 0.5 * (a[0] * b[1] - a[1] * b[0])
        # ∫_T ||x-p||² dA = (|A|/6)(|a|² + |b|² + a·b) — but use signed A so
        # contributions cancel correctly for non-convex/orientation issues
        integrand = (a @ a + b @ b + a @ b) / 6.0
        energy += tri_A * integrand
    return abs(energy)


# =============================================================================
# UNIFIED ITERATION (DIVI or LLOYD)
# =============================================================================

def voronoi_iteration_torus(m=25, Lx=10.0, Ly=10.0, eps=1e-8, max_iter=100,
                            seed=42, method='divi', X0=None, verbose=True):
    """
    method = 'divi'  : x_i^new = mean of cell vertices  (Differentiation-Induced)
    method = 'lloyd' : x_i^new = true centroid of cell  (classic Lloyd / CVT)
    """
    if X0 is None:
        np.random.seed(seed)
        X = np.c_[np.random.uniform(0, Lx, m), np.random.uniform(0, Ly, m)]
    else:
        X = X0.copy()
        m = len(X)

    history = [X.copy()]
    errors  = []
    energies = []
    offset  = get_original_index_offset(m)

    for it in range(1, max_iter + 1):
        vor = voronoi_on_torus(X, Lx, Ly)
        X_new = X.copy()
        E_total = 0.0

        for i in range(m):
            global_idx = offset + i
            region_idx = vor.point_region[global_idx]
            region = vor.regions[region_idx]

            if -1 in region or len(region) == 0:
                continue

            V = vor.vertices[region]
            if len(V) < 3:
                continue

            # Voronoi (CVT) energy contribution — same definition for both methods
            # so the comparison is fair
            E_total += polygon_second_moment(V, X[i])

            if method == 'divi':
                # Average of vertices
                c = V.mean(axis=0)
            elif method == 'lloyd':
                # True polygon centroid (area-weighted)
                _, c = polygon_area_centroid(V)
            else:
                raise ValueError(f"Unknown method: {method}")

            X_new[i] = [c[0] % Lx, c[1] % Ly]

        # Toroidal displacement
        diff = X_new - X
        diff[:, 0] -= Lx * np.round(diff[:, 0] / Lx)
        diff[:, 1] -= Ly * np.round(diff[:, 1] / Ly)
        max_move = np.max(np.linalg.norm(diff, axis=1))

        errors.append(max_move)
        energies.append(E_total)
        X = X_new
        history.append(X.copy())

        if verbose:
            print(f"  [{method:5s}] iter {it:3d}  |  max move = {max_move:.3e}"
                  f"  |  E = {E_total:.6f}")

        if max_move < eps:
            if verbose:
                print(f"  ✓ [{method}] Converged at iteration {it}")
            break

    return X, history, errors, energies


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_voronoi(X, Lx, Ly, ax, color='b', title=''):
    vor = voronoi_on_torus(X, Lx, Ly)
    margin = 0.01
    for simplex in vor.ridge_vertices:
        if -1 in simplex:
            continue
        v0, v1 = vor.vertices[simplex]
        if (-margin <= v0[0] <= Lx+margin and -margin <= v0[1] <= Ly+margin and
            -margin <= v1[0] <= Lx+margin and -margin <= v1[1] <= Ly+margin):
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], color=color, lw=0.6, alpha=0.5)

    ax.scatter(X[:, 0], X[:, 1], c=color, s=20, zorder=5,
               edgecolors='k', linewidths=0.3)
    rect = plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor='black',
                         lw=1.2, ls='--')
    ax.add_patch(rect)
    ax.set(xlim=(-0.3, Lx+0.3), ylim=(-0.3, Ly+0.3), aspect='equal', title=title)


def visualize_side_by_side(hist_divi, hist_lloyd, Lx, Ly, n_snap=4):
    """Two rows: top=DIVI, bottom=Lloyd, columns = matched iterations."""
    n_iters = min(len(hist_divi), len(hist_lloyd))
    idxs = np.linspace(0, n_iters - 1, n_snap, dtype=int)

    fig, axes = plt.subplots(2, n_snap, figsize=(4.2 * n_snap, 8.4))
    for k, i in enumerate(idxs):
        draw_voronoi(hist_divi[i],  Lx, Ly, axes[0, k], color='tab:red',
                     title=f'DIVI   iter {i}')
        draw_voronoi(hist_lloyd[i], Lx, Ly, axes[1, k], color='tab:blue',
                     title=f'Lloyd  iter {i}')
    fig.suptitle('DIVI (vertex average)  vs  classic Lloyd (true centroid)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('compare_snapshots.png', dpi=150)
    plt.show()


def visualize_overlay_final(X_divi, X_lloyd, Lx, Ly):
    """Overlay both final configurations on a single axis."""
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_voronoi(X_divi,  Lx, Ly, ax, color='tab:red',  title='Final: DIVI (red) vs Lloyd (blue)')
    # Re-draw Lloyd over it (lines will be different colors)
    draw_voronoi(X_lloyd, Lx, Ly, ax, color='tab:blue', title='Final: DIVI (red) vs Lloyd (blue)')
    plt.tight_layout()
    plt.savefig('compare_overlay_final.png', dpi=150)
    plt.show()


def plot_convergence_compare(err_divi, err_lloyd):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(err_divi)+1),  err_divi,  'o-', ms=3,
                color='tab:red',  label='DIVI (vertex avg)')
    ax.semilogy(range(1, len(err_lloyd)+1), err_lloyd, 's-', ms=3,
                color='tab:blue', label='Lloyd (true centroid)')
    ax.set(xlabel='Iteration',
           ylabel=r'$\max_i \|x_i^{\mathrm{new}} - x_i\|$',
           title='Convergence comparison (Torus)')
    ax.legend()
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig('compare_convergence.png', dpi=150)
    plt.show()


def plot_energy_compare(E_divi, E_lloyd):
    """Voronoi (CVT) energy E = Σ_i ∫_{V_i} ||x - x_i||² dA vs iteration."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(E_divi)+1),  E_divi,  'o-', ms=3,
            color='tab:red',  label='DIVI (vertex avg)')
    ax.plot(range(1, len(E_lloyd)+1), E_lloyd, 's-', ms=3,
            color='tab:blue', label='Lloyd (true centroid)')
    ax.set(xlabel='Iteration',
           ylabel=r'CVT energy $\sum_i \int_{V_i} \|x - x_i\|^2\,dA$',
           title='Voronoi energy vs iteration')
    ax.legend()
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig('compare_energy.png', dpi=150)
    plt.show()

    # Also a log-scale version of (E - E_min) — often more revealing
    E_min = min(min(E_divi), min(E_lloyd))
    fig, ax = plt.subplots(figsize=(8, 4))
    gap_d = np.array(E_divi)  - E_min + 1e-16
    gap_l = np.array(E_lloyd) - E_min + 1e-16
    ax.semilogy(range(1, len(gap_d)+1), gap_d, 'o-', ms=3,
                color='tab:red',  label='DIVI')
    ax.semilogy(range(1, len(gap_l)+1), gap_l, 's-', ms=3,
                color='tab:blue', label='Lloyd')
    ax.set(xlabel='Iteration',
           ylabel=r'$E - E_{\min}$ (log)',
           title='Energy gap to best observed minimum')
    ax.legend()
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig('compare_energy_gap.png', dpi=150)
    plt.show()


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  DIVI vs classic Lloyd  on a flat torus [0,L)×[0,L)")
    print("=" * 60)

    Lx, Ly = 10.0, 10.0
    M = 50
    SEED = 60
    EPS = 1e-5
    MAX_IT = 200

    # ---- Identical initial conditions for both methods ----
    rng = np.random.RandomState(SEED)
    X0 = np.c_[rng.uniform(0, Lx, M), rng.uniform(0, Ly, M)]

    print("\n--- Running DIVI (vertex average) ---")
    X_d, hist_d, err_d, E_d = voronoi_iteration_torus(
        Lx=Lx, Ly=Ly, eps=EPS, max_iter=MAX_IT,
        method='divi', X0=X0)

    print("\n--- Running classic Lloyd (true centroid) ---")
    X_l, hist_l, err_l, E_l = voronoi_iteration_torus(
        Lx=Lx, Ly=Ly, eps=EPS, max_iter=MAX_IT,
        method='lloyd', X0=X0)

    # ---- Plots ----
    visualize_side_by_side(hist_d, hist_l, Lx, Ly, n_snap=4)
    visualize_overlay_final(X_d, X_l, Lx, Ly)
    plot_convergence_compare(err_d, err_l)
    plot_energy_compare(E_d, E_l)

    print("\n" + "=" * 60)
    print(f"  DIVI : {len(err_d):3d} iters,  final E = {E_d[-1]:.6f}")
    print(f"  Lloyd: {len(err_l):3d} iters,  final E = {E_l[-1]:.6f}")
    print("=" * 60)
    print("\nSaved figures:")
    print("  compare_snapshots.png      (rows: DIVI / Lloyd)")
    print("  compare_overlay_final.png  (final tessellations overlaid)")
    print("  compare_convergence.png    (max-move vs iter, semilog)")
    print("  compare_energy.png         (CVT energy vs iter)")
    print("  compare_energy_gap.png     (E - E_min, semilog)")