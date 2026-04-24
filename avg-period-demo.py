import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# =============================================================================
# TORUS VORONOI: Tile 3×3 copies to simulate periodic boundaries
# =============================================================================

def tile_points(pts, Lx, Ly):
    """
    Create 9 copies of the point set, shifted by all combinations
    of {-Lx, 0, +Lx} × {-Ly, 0, +Ly}.

    Returns:
        tiled : (9*N, 2) array
        The first N rows are the original points.
    """
    shifts = np.array([
        [dx, dy]
        for dx in [-Lx, 0, Lx]
        for dy in [-Ly, 0, Ly]
    ])
    copies = [pts + s for s in shifts]
    return np.vstack(copies)


def voronoi_on_torus(pts, Lx, Ly):
    """
    Compute Voronoi on a flat torus [0,Lx) × [0,Ly)
    by tiling, then extracting only the cells for the original N points.

    Returns:
        vor : scipy Voronoi object (on tiled points)
        N   : number of original generators
    """
    tiled = tile_points(pts, Lx, Ly)
    # The original points are in the block corresponding to shift (0,0).
    # In our ordering: shifts are (-L,-L),(-L,0),(-L,+L),(0,-L),(0,0),...
    # (0,0) is index 4 in the 9 shifts → rows [4*N : 5*N]
    vor = Voronoi(tiled)
    return vor


def get_original_index_offset(N):
    """The (0,0) copy sits at index 4 in our 3×3 tiling."""
    return 4 * N


# =============================================================================
# ALGORITHM: Differentiation-Induced Voronoi Iteration on a Torus
# =============================================================================

def divi_torus(m=25, Lx=10.0, Ly=10.0, eps=1e-8, max_iter=100, seed=42):
    """
    Differentiation-Induced Voronoi Iteration on a flat torus [0,Lx) × [0,Ly).

    1. Initialize generators    x₁, ..., xₘ ∈ [0,Lx) × [0,Ly)
    2. Tile 3×3 copies          (periodic boundary)
    3. Compute Voronoi diagram  on tiled points
    4. Extract vertices         {vᵢ₁, ..., vᵢₙᵢ} of each original cell
    5. Collapse via centroid    cᵢ = (1/nᵢ) Σⱼ vᵢⱼ
    6. Update generators        xᵢⁿᵉʷ = cᵢ  (mod Lx, mod Ly)
    7. Repeat until             maxᵢ ‖xᵢⁿᵉʷ − xᵢ‖ < ε
    """
    np.random.seed(seed)

    # Step 1: Initialize generators
    X = np.c_[
        np.random.uniform(0, Lx, m),
        np.random.uniform(0, Ly, m),
    ]

    history = [X.copy()]
    errors  = []
    offset  = get_original_index_offset(m)

    for it in range(1, max_iter + 1):

        # Steps 2-3: Tile and compute Voronoi
        vor = voronoi_on_torus(X, Lx, Ly)

        X_new = X.copy()
        for i in range(m):
            # The original point i is at index (offset + i) in the tiled array
            global_idx = offset + i
            region_idx = vor.point_region[global_idx]
            region = vor.regions[region_idx]

            # With tiling, no cell should be unbounded
            if -1 in region or len(region) == 0:
                continue

            # Step 4: Extract vertices
            V = vor.vertices[region]

            if len(V) < 3:
                continue

            # Step 5: Collapse via differentiation (centroid of vertices)
            #         cᵢ = (1/nᵢ) Σⱼ vᵢⱼ
            c = V.mean(axis=0)

            # Step 6: Wrap back onto torus
            X_new[i] = [c[0] % Lx, c[1] % Ly]

        # Step 7: Check convergence
        # Use toroidal distance for correct movement measurement
        diff = X_new - X
        # Wrap differences to [-L/2, L/2]
        diff[:, 0] -= Lx * np.round(diff[:, 0] / Lx)
        diff[:, 1] -= Ly * np.round(diff[:, 1] / Ly)
        max_move = np.max(np.linalg.norm(diff, axis=1))

        errors.append(max_move)
        X = X_new
        history.append(X.copy())

        print(f"  iter {it:3d}  |  max ‖xⁿᵉʷ − x‖ = {max_move:.10f}")

        if max_move < eps:
            print(f"  ✓ Converged at iteration {it}")
            break

    return X, history, errors


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_torus_voronoi(X, Lx, Ly, ax, title=''):
    """Plot Voronoi on the torus, clipped to the fundamental domain."""
    vor = voronoi_on_torus(X, Lx, Ly)
    offset = get_original_index_offset(len(X))

    # Draw edges that fall within [0,Lx] × [0,Ly]
    margin = 0.01
    for simplex in vor.ridge_vertices:
        if -1 in simplex:
            continue
        v0, v1 = vor.vertices[simplex]
        # Only draw if both endpoints are in the fundamental domain (with tiny margin)
        if (-margin <= v0[0] <= Lx+margin and -margin <= v0[1] <= Ly+margin and
            -margin <= v1[0] <= Lx+margin and -margin <= v1[1] <= Ly+margin):
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], 'b-', lw=0.6, alpha=0.5)

    ax.scatter(X[:, 0], X[:, 1], c='red', s=20, zorder=5, edgecolors='k', linewidths=0.3)

    # Draw the fundamental domain boundary
    rect = plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor='black', lw=1.5, ls='--')
    ax.add_patch(rect)

    ax.set(xlim=(-0.3, Lx+0.3), ylim=(-0.3, Ly+0.3), aspect='equal', title=title)


def visualize(history, errors, Lx, Ly):
    # --- Snapshots ---
    n = min(8, len(history))
    idxs = np.linspace(0, len(history)-1, n, dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, i in zip(axes.flat, idxs):
        plot_torus_voronoi(history[i], Lx, Ly, ax, title=f'iter {i}')
    fig.suptitle('Differentiation-Induced Voronoi Iteration (Torus)', fontsize=14)
    plt.tight_layout()
    plt.savefig('divi_torus_snapshots.png', dpi=150)
    plt.show()

    # --- Convergence ---
    if len(errors) > 1 and any(e > 0 for e in errors):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.semilogy(range(1, len(errors)+1), errors, 'o-', ms=3)
        ax.axhline(1e-8, color='gray', ls='--', lw=.8, label='ε = 1e-8')
        ax.set(xlabel='Iteration',
               ylabel=r'$\max_i \|x_i^{\mathrm{new}} - x_i\|$',
               title='DIVI Convergence (Torus)')
        ax.legend()
        ax.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig('divi_torus_convergence.png', dpi=150)
        plt.show()


def plot_trajectories_torus(history, Lx, Ly):
    """Plot trajectories on the torus (no wrapping artifacts)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    history = np.array(history)  # (n_iter+1, m, 2)

    for i in range(history.shape[1]):
        traj = history[:, i, :]
        # Break trajectory at wrap-around points to avoid lines across the domain
        dx = np.abs(np.diff(traj[:, 0]))
        dy = np.abs(np.diff(traj[:, 1]))
        breaks = np.where((dx > Lx/2) | (dy > Ly/2))[0]

        # Split into segments
        segments = np.split(np.arange(len(traj)), breaks + 1)
        for seg in segments:
            if len(seg) > 1:
                ax.plot(traj[seg, 0], traj[seg, 1], 'o-', ms=2, lw=0.8, alpha=0.6)
            else:
                ax.plot(traj[seg, 0], traj[seg, 1], 'o', ms=2, alpha=0.6)

    rect = plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor='black', lw=1.5, ls='--')
    ax.add_patch(rect)
    ax.set(xlim=(-0.3, Lx+0.3), ylim=(-0.3, Ly+0.3), aspect='equal',
           title='Trajectories (Torus)')
    plt.tight_layout()
    plt.savefig('divi_torus_trajectories.png', dpi=150)
    plt.show()


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    print("=" * 55)
    print("  Differentiation-Induced Voronoi Iteration (DIVI)")
    print("  on a flat torus [0,L)×[0,L)  — no boundaries!")
    print("=" * 55)
    print("  cᵢ = (1/nᵢ) Σⱼ vᵢⱼ   →   xᵢⁿᵉʷ = cᵢ mod L")
    print()

    Lx, Ly = 10.0, 10.0
    M = 40

    X, history, errors = divi_torus(m=M, Lx=Lx, Ly=Ly, eps=1e-5, max_iter=100, seed=42)

    visualize(history, errors, Lx, Ly)
    plot_trajectories_torus(history, Lx, Ly)

    print("\nSaved:")
    print("  divi_torus_snapshots.png")
    print("  divi_torus_convergence.png")
    print("  divi_torus_trajectories.png")