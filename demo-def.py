import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import matplotlib.cm as cm
from matplotlib.patches import Polygon


def get_voronoi_cell_vertices(points, cell_index):
    """Get vertices of a specific Voronoi cell."""
    vor = Voronoi(points)
    region_index = vor.point_region[cell_index]
    region = vor.regions[region_index]

    if -1 in region:
        region = [v for v in region if v != -1]

    if len(region) == 0:
        return None, vor

    vertices = vor.vertices[region]
    return vertices, vor


def vertices_to_complex(vertices):
    return vertices[:, 0] + 1j * vertices[:, 1]


def complex_to_vertices(z_array):
    return np.column_stack([z_array.real, z_array.imag])


def build_companion_matrix(roots):
    """Build companion matrix whose eigenvalues are the given roots."""
    coeffs = np.poly(roots)
    n = len(roots)
    C = np.zeros((n, n), dtype=complex)
    for i in range(1, n):
        C[i, i - 1] = 1.0
    for i in range(n):
        C[i, n - 1] = -coeffs[n - i]
    return C


def monomial_vector(z, n):
    """Build (1, z, z^2, ..., z^{n-1})^T"""
    return np.array([z ** k for k in range(n)], dtype=complex)


def monomial_vector_derivative(z, n, order=1):
    """
    k-th derivative of (1, z, z^2, ..., z^{n-1})^T with respect to z.

    d^k/dz^k [z^m] = m!/(m-k)! * z^{m-k}  if m >= k, else 0
    """
    result = np.zeros(n, dtype=complex)
    for m in range(n):
        if m >= order:
            coeff = 1.0
            for j in range(order):
                coeff *= (m - j)
            result[m] = coeff * z ** (m - order)
        else:
            result[m] = 0.0
    return result


def build_differentiation_matrix(n):
    """
    Build matrix D such that D @ v(z) = v'(z)

    D maps [1, z, z^2, ..., z^{n-1}]^T -> [0, 1, 2z, ..., (n-1)z^{n-2}]^T

    But wait — D can't be constant because v'(z) depends on z.

    However, in the monomial basis, the "shift" structure is:
    D_ij = j * delta_{i, j-1}

    This gives: D @ [c_0, c_1, c_2, ...]^T = [c_1, 2*c_2, 3*c_3, ...]^T
    which is differentiation of the polynomial c_0 + c_1*z + c_2*z^2 + ...
    """
    D = np.zeros((n, n), dtype=complex)
    for j in range(1, n):
        D[j - 1, j] = j
    return D


def analyze_Cv_derivative(C, z_values, n_derivatives=None):
    """
    For each eigenvalue z_i (vertex), compute:
      C @ v(z_i), C @ v'(z_i), C @ v''(z_i), ...

    And analyze what happens.
    """
    n = C.shape[0]
    if n_derivatives is None:
        n_derivatives = n

    results = {}

    for idx, z in enumerate(z_values):
        results[idx] = {
            'z': z,
            'products': [],
            'norms': []
        }

        for k in range(n_derivatives):
            vk = monomial_vector_derivative(z, n, order=k)
            product = C @ vk
            results[idx]['products'].append(product)
            results[idx]['norms'].append(np.linalg.norm(product))

    return results


def analyze_polynomial_through_matrix(C, z_eval_points, n_derivatives=None):
    """
    The companion matrix encodes p(z) = det(zI - C).

    C @ v(z) can be related to polynomial evaluation.

    Let's look at what C @ v^(k)(z) tells us about
    the k-th derivative of the polynomial evaluated at various points.
    """
    n = C.shape[0]
    if n_derivatives is None:
        n_derivatives = n

    D = build_differentiation_matrix(n)

    print("\nDifferentiation matrix D:")
    print(D)
    print(f"\nCompanion matrix C ({n}x{n}):")
    print(np.round(C, 4))
    print(f"\nProduct D @ C (differentiation acting on companion):")
    print(np.round(D @ C, 4))
    print(f"\nProduct C @ D (companion acting on differentiation):")
    print(np.round(C @ D, 4))
    print(f"\nCommutator [D, C] = DC - CD:")
    print(np.round(D @ C - C @ D, 4))

    return D


def iterated_DC_product(C, max_iter=None):
    """
    What if we look at the matrix product D^k @ C or C @ D^k?
    Or the iterated commutator?

    The idea: D is differentiation, C encodes the polynomial.
    Their interaction might drive eigenvalues toward center.
    """
    n = C.shape[0]
    if max_iter is None:
        max_iter = n - 1

    D = build_differentiation_matrix(n)

    all_eigenvalues = [np.linalg.eigvals(C)]

    print(f"\nIteration 0: eigenvalues of C")
    print(f"  {all_eigenvalues[0]}")
    print(f"  centroid: {np.mean(all_eigenvalues[0]):.6f}")

    current = C.copy()

    for k in range(1, max_iter + 1):
        # Strategy 1: C_{k+1} = D @ C_k (left multiply by D)
        # Strategy 2: C_{k+1} = C_k @ D
        # Strategy 3: C_{k+1} = [D, C_k] = D @ C_k - C_k @ D (commutator)
        # Strategy 4: C_{k+1} = (D @ C_k + C_k @ D) / 2 (anti-commutator / 2)

        # Let's try all and see which converges

        strategies = {
            'DC': D @ current,
            'CD': current @ D,
            'commutator': D @ current - current @ D,
            'anti_commutator': (D @ current + current @ D) / 2
        }

        print(f"\nIteration {k}:")
        for name, M in strategies.items():
            eigs = np.linalg.eigvals(M)
            finite_eigs = eigs[np.isfinite(eigs)]
            if len(finite_eigs) > 0:
                centroid = np.mean(finite_eigs)
                spread = np.std(finite_eigs)
                print(f"  {name:20s}: centroid={centroid:.6f}, spread={spread:.6f}")
                print(f"  {'':20s}  eigs={np.round(finite_eigs, 4)}")

        # Use commutator as default progression
        current = D @ current - current @ D
        eigs = np.linalg.eigvals(current)
        all_eigenvalues.append(eigs)

    return all_eigenvalues


def evaluate_Cv_on_grid(C, grid_z, derivative_order=0):
    """
    Evaluate C @ v^(k)(z) on a grid of z values.
    Returns the norm of the result at each grid point.

    This creates a "landscape" showing where C @ v^(k)(z) is small
    (i.e., near eigenvalues for k=0).
    """
    n = C.shape[0]
    norms = np.zeros(len(grid_z), dtype=float)

    for i, z in enumerate(grid_z):
        vk = monomial_vector_derivative(z, n, order=derivative_order)
        product = C @ vk
        # For k=0: C@v(z) = z*v(z) at eigenvalues, so ||Cv - zv|| = 0
        # Let's measure ||Cv(z) - z*v(z)|| to find eigenvalues
        if derivative_order == 0:
            residual = product - z * monomial_vector(z, n)
            norms[i] = np.linalg.norm(residual)
        else:
            norms[i] = np.linalg.norm(product)

    return norms


def full_derivative_analysis(C, z_vertices):
    """
    Complete analysis of C @ v^(k)(z) for all derivative orders.
    """
    n = C.shape[0]
    D = build_differentiation_matrix(n)

    print("=" * 70)
    print("ANALYSIS: d/dz [C @ v(z)] = C @ v'(z) = C @ D @ v(z)")
    print("=" * 70)

    # Key relationship: v'(z) can be expressed as D_z @ v(z)
    # where D_z is z-dependent. But in coefficient space, D is constant.

    # The polynomial p(z) has coefficients stored in companion matrix.
    # If we write p(z) = c^T @ v(z) where c is the last row relationship,
    # then p'(z) = c^T @ v'(z) = c^T @ D_z @ v(z)

    # Let's look at C^k @ v(z) evaluated at eigenvalues
    print("\n--- C^k @ v(z) at eigenvalues (should give z^k * v(z)) ---")
    for idx, z in enumerate(z_vertices):
        v = monomial_vector(z, n)
        print(f"\nVertex z_{idx} = {z:.4f}:")
        for k in range(min(4, n)):
            Ck_v = np.linalg.matrix_power(C, k) @ v
            expected = (z ** k) * v
            match = np.allclose(Ck_v, expected)
            print(f"  C^{k} @ v(z): matches z^{k}*v(z)? {match}")

    # Now the key: what does C @ v'(z) give us at eigenvalues?
    print("\n--- C @ v^(k)(z) at eigenvalues ---")
    for idx, z in enumerate(z_vertices):
        print(f"\nVertex z_{idx} = {z:.4f}:")
        for k in range(n):
            vk = monomial_vector_derivative(z, n, order=k)
            Cvk = C @ vk
            norm_val = np.linalg.norm(Cvk)
            print(f"  C @ v^({k})(z): norm = {norm_val:.6f}")
            if k < 3:
                print(f"  {'':14s} value = {np.round(Cvk, 4)}")

    return D


def track_Cv_derivative_zeros(C, z_vertices, grid_resolution=200):
    """
    For each derivative order k, find where ||C @ v^(k)(z)|| is minimized.
    These "zeros" should migrate toward the center.
    """
    n = C.shape[0]

    # Create grid in complex plane around the vertices
    x_min = z_vertices.real.min() - 1
    x_max = z_vertices.real.max() + 1
    y_min = z_vertices.imag.min() - 1
    y_max = z_vertices.imag.max() + 1

    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z_grid = X + 1j * Y

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    minima_locations = []

    for k in range(n):
        # Compute ||C @ v^(k)(z) - z * v^(k)(z)|| on grid (generalized residual)
        # For k=0 this finds eigenvalues
        # For k>0 this finds... what?
        norms = np.zeros_like(X, dtype=float)

        for i in range(grid_resolution):
            for j in range(grid_resolution):
                z = Z_grid[i, j]
                vk = monomial_vector_derivative(z, n, order=k)
                Cvk = C @ vk
                norms[i, j] = np.linalg.norm(Cvk)

        # Find minimum location
        min_idx = np.unravel_index(np.argmin(norms), norms.shape)
        min_z = Z_grid[min_idx]
        minima_locations.append(min_z)

        # Plot
        ax_top = axes[0, k] if n > 1 else axes[0]
        ax_bot = axes[1, k] if n > 1 else axes[1]

        # Heatmap
        im = ax_top.pcolormesh(X, Y, np.log10(norms + 1e-15), cmap='hot_r')
        ax_top.scatter(z_vertices.real, z_vertices.imag, c='cyan', s=50,
                       marker='o', label='vertices')
        ax_top.scatter(min_z.real, min_z.imag, c='lime', s=100,
                       marker='*', label=f'min')
        ax_top.scatter(np.mean(z_vertices).real, np.mean(z_vertices).imag,
                       c='white', s=100, marker='+', linewidths=2, label='centroid')
        ax_top.set_title(f"log₁₀ ||C @ v^({k})(z)||", fontsize=11)
        ax_top.set_aspect('equal')
        ax_top.legend(fontsize=7)
        plt.colorbar(im, ax=ax_top, fraction=0.046)

        # Also plot ||C @ v^(k)(z) - z * v^(k-1)(z)|| as alternative
        norms2 = np.zeros_like(X, dtype=float)
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                z = Z_grid[i, j]
                vk = monomial_vector_derivative(z, n, order=k)
                if k > 0:
                    vk_minus1 = monomial_vector_derivative(z, n, order=k - 1)
                    residual = C @ vk - z * vk_minus1
                else:
                    residual = C @ vk - z * vk
                norms2[i, j] = np.linalg.norm(residual)

        min_idx2 = np.unravel_index(np.argmin(norms2), norms2.shape)
        min_z2 = Z_grid[min_idx2]

        im2 = ax_bot.pcolormesh(X, Y, np.log10(norms2 + 1e-15), cmap='hot_r')
        ax_bot.scatter(z_vertices.real, z_vertices.imag, c='cyan', s=50, marker='o')
        ax_bot.scatter(min_z2.real, min_z2.imag, c='lime', s=100, marker='*')
        ax_bot.scatter(np.mean(z_vertices).real, np.mean(z_vertices).imag,
                       c='white', s=100, marker='+', linewidths=2)
        if k == 0:
            ax_bot.set_title(f"log₁₀ ||Cv(z) - zv(z)||", fontsize=11)
        else:
            ax_bot.set_title(f"log₁₀ ||Cv^({k}) - zv^({k - 1})||", fontsize=11)
        ax_bot.set_aspect('equal')
        plt.colorbar(im2, ax=ax_bot, fraction=0.046)

    plt.suptitle("Landscape of C @ v^(k)(z): Do minima converge to center?",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("Cv_derivative_landscape.png", dpi=150, bbox_inches='tight')
    plt.show()

    return minima_locations


def matrix_product_derivative_approach(z_vertices):
    """
    Main function: The full matrix-product derivative approach.

    Key idea:
    C @ v(z) = z * v(z)  at eigenvalues

    d/dz: C @ v'(z) = v(z) + z * v'(z)  (product rule on RHS)

    d²/dz²: C @ v''(z) = 2v'(z) + z * v''(z)

    In general: C @ v^(k)(z) = k * v^(k-1)(z) + z * v^(k)(z)

    So: (C - zI) @ v^(k)(z) = k * v^(k-1)(z)

    This is a RECURRENCE relating successive derivatives!
    """
    n = len(z_vertices)
    C = build_companion_matrix(z_vertices)

    print("=" * 70)
    print("MATRIX PRODUCT DERIVATIVE APPROACH")
    print("=" * 70)
    print(f"\nVertices (eigenvalues): {z_vertices}")
    print(f"Centroid: {np.mean(z_vertices):.6f}")
    print(f"\nCompanion matrix C:")
    print(np.round(C, 4))

    # The key identity at eigenvalue z_i:
    # C @ v(z_i) = z_i * v(z_i)          ... eigenvalue equation
    # C @ v'(z_i) = v(z_i) + z_i * v'(z_i)  ... differentiated
    # (C - z_i I) @ v'(z_i) = v(z_i)     ... rearranged!
    #
    # This means v'(z_i) = (C - z_i I)^{-1} @ v(z_i)  (if invertible)
    # i.e., it's a resolvent!

    print("\n--- Verifying the identity (C - zI) @ v'(z) = v(z) at eigenvalues ---")

    resolvent_points = []

    for idx, z in enumerate(z_vertices):
        v = monomial_vector(z, n)
        v1 = monomial_vector_derivative(z, n, order=1)

        lhs = (C - z * np.eye(n)) @ v1
        rhs = v

        print(f"\nz_{idx} = {z:.4f}:")
        print(f"  (C - zI) @ v'(z) = {np.round(lhs, 6)}")
        print(f"  v(z)              = {np.round(rhs, 6)}")
        print(f"  Match: {np.allclose(lhs, rhs, atol=1e-8)}")

        # Higher order: (C - zI) @ v^(k)(z) = k * v^(k-1)(z)
        for k in range(1, n):
            vk = monomial_vector_derivative(z, n, order=k)
            vk_minus1 = monomial_vector_derivative(z, n, order=k - 1)
            lhs_k = (C - z * np.eye(n)) @ vk
            rhs_k = k * vk_minus1
            match = np.allclose(lhs_k, rhs_k, atol=1e-8)
            print(f"  Order {k}: (C-zI)@v^({k}) = {k}*v^({k - 1})? {match}")

    # Now the big idea:
    # Define a "flow" using these derivatives
    # At each step, we're solving (C - zI) @ v^(k) = k * v^(k-1)
    # This involves the RESOLVENT (C - zI)^{-1}
    # The resolvent has poles at eigenvalues — it "feels" the spectral structure

    # What if we define a new matrix from this relationship?
    # C_new such that the relationship encodes the derivative structure?

    print("\n--- Resolvent-based analysis ---")
    print("At a generic point z₀ (not an eigenvalue):")

    z0 = np.mean(z_vertices)  # Try the centroid
    print(f"\nz₀ = centroid = {z0:.6f}")

    resolvent = np.linalg.inv(C - z0 * np.eye(n))
    print(f"Resolvent (C - z₀I)^{{-1}}:")
    print(np.round(resolvent, 4))
    print(f"Eigenvalues of resolvent: {np.round(np.linalg.eigvals(resolvent), 6)}")
    print(f"(Should be 1/(λ_i - z₀) for each eigenvalue λ_i)")

    for idx, z in enumerate(z_vertices):
        expected = 1.0 / (z - z0)
        print(f"  1/(z_{idx} - z₀) = 1/({z:.4f} - {z0:.4f}) = {expected:.6f}")

    # Differentiation matrix interaction
    D = build_differentiation_matrix(n)

    print(f"\n--- Matrix D (differentiation operator in monomial basis) ---")
    print(np.round(D.real, 4))

    # The product CD and DC
    CD = C @ D
    DC = D @ C

    print(f"\nEigenvalues of C:  {np.round(np.sort_complex(np.linalg.eigvals(C)), 4)}")
    print(f"Eigenvalues of CD: {np.round(np.sort_complex(np.linalg.eigvals(CD)), 4)}")
    print(f"Eigenvalues of DC: {np.round(np.sort_complex(np.linalg.eigvals(DC)), 4)}")

    commutator = DC - CD
    print(f"Eigenvalues of [D,C]: {np.round(np.sort_complex(np.linalg.eigvals(commutator)), 4)}")

    centroid_of_DC = np.mean(np.linalg.eigvals(DC))
    centroid_of_CD = np.mean(np.linalg.eigvals(CD))
    print(f"\nCentroid of DC eigenvalues: {centroid_of_DC:.6f}")
    print(f"Centroid of CD eigenvalues: {centroid_of_CD:.6f}")

    return C, D


def iterated_commutator_flow(C, D, n_steps=20):
    """
    Explore iterated commutator [D, C], [[D, C], D], etc.
    Does this drive eigenvalues toward center?
    """
    print("\n" + "=" * 70)
    print("ITERATED COMMUTATOR FLOW")
    print("=" * 70)

    original_centroid = np.mean(np.linalg.eigvals(C))
    print(f"Original eigenvalue centroid: {original_centroid:.6f}")

    current = C.copy()
    all_eigs = [np.linalg.eigvals(C)]

    for step in range(n_steps):
        comm = D @ current - current @ D

        eigs = np.linalg.eigvals(comm)
        finite_eigs = eigs[np.isfinite(eigs) & (np.abs(eigs) < 1e10)]

        if len(finite_eigs) == 0:
            print(f"Step {step + 1}: No finite eigenvalues. Stopping.")
            break

        centroid = np.mean(finite_eigs)
        spread = np.std(finite_eigs)
        print(f"Step {step + 1}: centroid={centroid:.6f}, spread={spread:.6f}, "
              f"n_eigs={len(finite_eigs)}")

        all_eigs.append(finite_eigs)
        current = comm

        if spread < 1e-10:
            print("Converged!")
            break

    return all_eigs


def plot_all_convergence(all_eigs_commutator, z_vertices, title="Commutator Flow"):
    """Plot the eigenvalue evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Complex plane
    ax1 = axes[0]
    colors = cm.viridis(np.linspace(0, 1, len(all_eigs_commutator)))

    for i, eigs in enumerate(all_eigs_commutator):
        ax1.scatter(eigs.real, eigs.imag, color=colors[i], s=80,
                    edgecolors='black', linewidth=0.5, zorder=5 + i,
                    label=f"step {i}" if i < 8 else None)

    centroid = np.mean(z_vertices)
    ax1.scatter(centroid.real, centroid.imag, color='red', s=200,
                marker='*', zorder=100, label='Vertex centroid')
    ax1.scatter(z_vertices.real, z_vertices.imag, color='cyan', s=100,
                marker='D', zorder=99, label='Original vertices', edgecolors='black')

    ax1.set_title(f"{title}\nEigenvalue Evolution", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Metrics
    ax2 = axes[1]
    spreads = [np.std(e) for e in all_eigs_commutator]
    centroids = [np.mean(e) for e in all_eigs_commutator]
    dists = [abs(np.mean(e) - centroid) for e in all_eigs_commutator]

    ax2.semilogy(range(len(spreads)), spreads, 'g-o', label='Spread')
    ax2.semilogy(range(len(dists)), dists, 'r-s', label='Dist to centroid')
    ax2.set_xlabel("Step")
    ax2.set_title("Convergence Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("matrix_product_derivative.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Generate Voronoi
    np.random.seed(42)
    n_points = 20
    points = np.random.rand(n_points, 2) * 10
    boundary = np.array([
        [-5, -5], [-5, 15], [15, -5], [15, 15],
        [5, -5], [5, 15], [-5, 5], [15, 5]
    ])
    all_points = np.vstack([points, boundary])

    cell_index = 5
    vertices, vor = get_voronoi_cell_vertices(all_points, cell_index)

    if vertices is None:
        print("Invalid cell")
        return

    z_vertices = vertices_to_complex(vertices)
    print(f"Cell {cell_index}: {len(vertices)} vertices")
    print(f"Vertices: {z_vertices}")
    print(f"Centroid: {np.mean(z_vertices):.6f}")

    # --- Approach 1: Full derivative analysis ---
    C, D = matrix_product_derivative_approach(z_vertices)

    # --- Approach 2: Full derivative analysis with identity verification ---
    full_derivative_analysis(C, z_vertices)

    # --- Approach 3: Iterated commutator flow ---
    all_eigs = iterated_commutator_flow(C, D, n_steps=15)
    plot_all_convergence(all_eigs, z_vertices, title="Commutator [D, C] Flow")

    # --- Approach 4: Landscape visualization ---
    print("\nComputing derivative landscapes (this may take a moment)...")
    minima = track_Cv_derivative_zeros(C, z_vertices, grid_resolution=100)

    print("\n--- Minima locations ---")
    centroid = np.mean(z_vertices)
    for k, m in enumerate(minima):
        print(f"  Order {k}: min at {m:.4f}, dist to centroid: {abs(m - centroid):.4f}")


if __name__ == "__main__":
    main()