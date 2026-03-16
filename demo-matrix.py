import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm


def get_voronoi_cell_vertices(points, cell_index):
    """
    Get the vertices of a specific Voronoi cell.
    """
    vor = Voronoi(points)
    region_index = vor.point_region[cell_index]
    region = vor.regions[region_index]

    if -1 in region:
        print(f"Warning: Cell {cell_index} is unbounded. Clipping vertices.")
        region = [v for v in region if v != -1]

    if len(region) == 0:
        return None, vor

    vertices = vor.vertices[region]
    return vertices, vor


def vertices_to_complex(vertices):
    """
    Convert 2D vertices to complex numbers: z = x + iy
    """
    return vertices[:, 0] + 1j * vertices[:, 1]


def complex_to_vertices(z_array):
    """
    Convert complex numbers back to 2D vertices.
    """
    return np.column_stack([z_array.real, z_array.imag])


def build_companion_matrix(roots):
    """
    Build a companion matrix whose eigenvalues are the given roots.
    The characteristic polynomial is: p(z) = prod(z - r_i)
    We expand this to get coefficients, then build the companion matrix.
    """
    # Build polynomial from roots: coefficients in descending order
    coeffs = np.poly(roots)  # Leading coeff is 1 (monic)
    n = len(roots)

    # Companion matrix (standard form)
    # The last row contains -c0, -c1, ..., -c_{n-1}
    C = np.zeros((n, n), dtype=complex)

    # Sub-diagonal of ones
    for i in range(1, n):
        C[i, i - 1] = 1.0

    # Last column contains the negated coefficients (excluding leading 1)
    for i in range(n):
        C[i, n - 1] = -coeffs[n - i]

    return C


def matrix_derivative(C):
    """
    Compute a 'derivative' of the companion matrix.

    Strategy: We interpret the companion matrix through its characteristic polynomial.
    The derivative of the characteristic polynomial p(z) has roots that are
    related to the "critical points" — by Gauss-Lucas theorem, they lie in the
    convex hull of the original roots.

    So: extract char poly -> differentiate -> build new companion matrix from
    the derivative's roots.
    """
    # Get characteristic polynomial coefficients
    coeffs = np.poly(np.linalg.eigvals(C))

    # Differentiate the polynomial
    d_coeffs = np.polyder(coeffs)

    # Find roots of derivative
    d_roots = np.roots(d_coeffs)

    return d_roots


def iterative_derivative_convergence(vertices_complex, max_iter=None):
    """
    Iteratively take derivatives of the characteristic polynomial
    (via companion matrix) until we reach a single point.

    By the Gauss-Lucas theorem, roots of the derivative lie within
    the convex hull of the original roots. Repeated differentiation
    should converge toward a single point.

    Returns list of root sets at each iteration.
    """
    n = len(vertices_complex)
    if max_iter is None:
        max_iter = n - 1  # After n-1 derivatives, we have 1 root

    all_roots = [vertices_complex.copy()]
    current_roots = vertices_complex.copy()

    for iteration in range(max_iter):
        if len(current_roots) <= 1:
            break

        # Build companion matrix
        C = build_companion_matrix(current_roots)

        # Take derivative (get roots of derivative of char poly)
        new_roots = matrix_derivative(C)

        all_roots.append(new_roots)
        current_roots = new_roots

        print(f"Iteration {iteration + 1}: {len(new_roots)} roots")
        print(f"  Roots: {new_roots}")
        print(f"  Centroid of roots: {np.mean(new_roots):.6f}")

    return all_roots


def direct_polynomial_derivative_chain(vertices_complex):
    """
    Alternative approach: directly work with polynomials.
    Build p(z) = prod(z - v_i), then repeatedly differentiate.
    """
    roots = vertices_complex.copy()
    coeffs = np.poly(roots)

    all_roots = [roots]
    all_coeffs = [coeffs]

    current_coeffs = coeffs

    iteration = 0
    while len(current_coeffs) > 2:  # degree > 0
        current_coeffs = np.polyder(current_coeffs)
        current_roots = np.roots(current_coeffs)

        all_roots.append(current_roots)
        all_coeffs.append(current_coeffs)

        iteration += 1

    # The last "root" is the final convergence point
    if len(current_coeffs) == 2:
        final_root = -current_coeffs[1] / current_coeffs[0]
        all_roots.append(np.array([final_root]))

    return all_roots


def analyze_convergence(all_roots, cell_centroid):
    """
    Analyze how the derivative roots converge relative to the cell centroid.
    """
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    print(f"Cell centroid: {cell_centroid:.6f}")
    print()

    for i, roots in enumerate(all_roots):
        centroid_of_roots = np.mean(roots)
        spread = np.std(roots)
        dist_to_centroid = abs(centroid_of_roots - cell_centroid)

        print(f"Derivative order {i}:")
        print(f"  Number of roots: {len(roots)}")
        print(f"  Mean (centroid of roots): {centroid_of_roots:.6f}")
        print(f"  Spread (std dev): {spread:.6f}")
        print(f"  Distance to cell centroid: {dist_to_centroid:.6f}")
        print()

    final_point = np.mean(all_roots[-1])
    print(f"Final convergence point: {final_point:.6f}")
    print(f"Cell centroid:           {cell_centroid:.6f}")
    print(f"Distance:                {abs(final_point - cell_centroid):.6f}")

    # Also compare to mean of original vertices
    original_mean = np.mean(all_roots[0])
    print(f"\nMean of original vertices: {original_mean:.6f}")
    print(f"Distance final->vertex_mean: {abs(final_point - original_mean):.6f}")


def plot_derivative_convergence(all_roots, cell_centroid, original_vertices, title=""):
    """
    Visualize the convergence of derivative roots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- Plot 1: All derivative levels overlaid ---
    ax1 = axes[0]
    colors = cm.viridis(np.linspace(0, 1, len(all_roots)))

    for i, roots in enumerate(all_roots):
        pts = complex_to_vertices(roots)
        ax1.scatter(pts[:, 0], pts[:, 1], color=colors[i], s=100 - i * 5,
                    zorder=5 + i, label=f"d^{i} ({len(roots)} pts)",
                    edgecolors='black', linewidth=0.5)

    # Plot centroid
    ax1.scatter(cell_centroid.real, cell_centroid.imag, color='red', s=200,
                marker='*', zorder=100, label='Cell centroid', edgecolors='black')

    # Draw original cell polygon
    if original_vertices is not None:
        poly = Polygon(original_vertices, fill=False, edgecolor='black',
                       linewidth=2, linestyle='--')
        ax1.add_patch(poly)

    ax1.set_title(f"Derivative Root Convergence\n{title}", fontsize=12)
    ax1.legend(fontsize=8, loc='best')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Trajectory of centroids ---
    ax2 = axes[1]
    centroids = [np.mean(roots) for roots in all_roots]
    centroids_2d = complex_to_vertices(np.array(centroids))

    ax2.plot(centroids_2d[:, 0], centroids_2d[:, 1], 'b-o', markersize=8,
             linewidth=2, label='Centroid trajectory')

    # Mark start and end
    ax2.scatter(centroids_2d[0, 0], centroids_2d[0, 1], color='green', s=150,
                marker='s', zorder=10, label='Start (d^0 centroid)')
    ax2.scatter(centroids_2d[-1, 0], centroids_2d[-1, 1], color='blue', s=150,
                marker='D', zorder=10, label='End (final point)')
    ax2.scatter(cell_centroid.real, cell_centroid.imag, color='red', s=200,
                marker='*', zorder=10, label='Cell centroid')

    if original_vertices is not None:
        poly = Polygon(original_vertices, fill=False, edgecolor='gray',
                       linewidth=1, linestyle='--')
        ax2.add_patch(poly)

    ax2.set_title("Centroid Trajectory Through Derivatives", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Spread and distance metrics ---
    ax3 = axes[2]
    spreads = [np.std(roots) for roots in all_roots]
    dists = [abs(np.mean(roots) - cell_centroid) for roots in all_roots]

    iterations = range(len(all_roots))
    ax3.plot(iterations, spreads, 'g-o', label='Spread (std dev)', linewidth=2)
    ax3.plot(iterations, dists, 'r-s', label='Dist to centroid', linewidth=2)
    ax3.set_xlabel("Derivative Order", fontsize=12)
    ax3.set_ylabel("Value", fontsize=12)
    ax3.set_title("Convergence Metrics", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig("voronoi_derivative_convergence.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_companion_matrix_evolution(all_roots):
    """
    Visualize the companion matrices at each derivative level.
    """
    n_matrices = min(len(all_roots), 6)
    fig, axes = plt.subplots(1, n_matrices, figsize=(4 * n_matrices, 4))

    if n_matrices == 1:
        axes = [axes]

    for i in range(n_matrices):
        roots = all_roots[i]
        if len(roots) < 1:
            break
        C = build_companion_matrix(roots)

        ax = axes[i]
        im = ax.imshow(np.abs(C), cmap='hot', interpolation='nearest')
        ax.set_title(f"d^{i} |C| ({len(roots)}x{len(roots)})", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle("Companion Matrix Magnitude Evolution", fontsize=14)
    plt.tight_layout()
    plt.savefig("companion_matrix_evolution.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_eigenvalue_spectrum(all_roots):
    """
    Plot eigenvalue spectrum at each level on the complex plane.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    colors = cm.plasma(np.linspace(0, 1, len(all_roots)))

    for i, roots in enumerate(all_roots):
        ax.scatter(roots.real, roots.imag, color=colors[i], s=150 - i * 10,
                   edgecolors='black', linewidth=0.5, zorder=5 + i,
                   label=f"d^{i}: {len(roots)} eigenvalues")

        # Connect to next level's roots with light lines
        if i < len(all_roots) - 1:
            next_roots = all_roots[i + 1]
            for r in roots:
                for nr in next_roots:
                    ax.plot([r.real, nr.real], [r.imag, nr.imag],
                            color=colors[i], alpha=0.1, linewidth=0.5)

    ax.set_xlabel("Real", fontsize=12)
    ax.set_ylabel("Imaginary", fontsize=12)
    ax.set_title("Eigenvalue Spectrum Through Derivatives\n(Gauss-Lucas Convergence)",
                 fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eigenvalue_spectrum.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("VORONOI CELL → COMPANION MATRIX → DERIVATIVE CONVERGENCE")
    print("=" * 70)
    print()
    print("Theory: By the Gauss-Lucas theorem, the roots of the derivative")
    print("of a polynomial lie within the convex hull of the original roots.")
    print("Repeated differentiation should converge toward a single point.")
    print()

    # --- Generate Voronoi diagram ---
    np.random.seed(42)
    n_points = 20
    points = np.random.rand(n_points, 2) * 10

    # Add boundary points to ensure bounded cells
    boundary = np.array([
        [-5, -5], [-5, 15], [15, -5], [15, 15],
        [5, -5], [5, 15], [-5, 5], [15, 5]
    ])
    all_points = np.vstack([points, boundary])

    # Choose a cell to analyze (pick one near center for bounded cell)
    cell_index = 5  # Arbitrary choice

    vertices, vor = get_voronoi_cell_vertices(all_points, cell_index)

    if vertices is None:
        print("Could not get valid cell. Try different cell_index.")
        return

    print(f"Selected cell {cell_index} with {len(vertices)} vertices:")
    for i, v in enumerate(vertices):
        print(f"  v_{i}: ({v[0]:.4f}, {v[1]:.4f})")

    # Compute cell centroid (geometric center)
    cell_centroid_2d = np.mean(vertices, axis=0)
    cell_centroid = cell_centroid_2d[0] + 1j * cell_centroid_2d[1]
    print(f"\nCell centroid: ({cell_centroid_2d[0]:.4f}, {cell_centroid_2d[1]:.4f})")
    print(f"Generator point: ({all_points[cell_index][0]:.4f}, {all_points[cell_index][1]:.4f})")

    # --- Convert vertices to complex numbers ---
    z_vertices = vertices_to_complex(vertices)
    print(f"\nComplex vertices: {z_vertices}")

    # --- Build initial companion matrix ---
    C0 = build_companion_matrix(z_vertices)
    print(f"\nInitial companion matrix shape: {C0.shape}")
    print(
        f"Eigenvalues match vertices: {np.allclose(sorted(np.linalg.eigvals(C0), key=abs), sorted(z_vertices, key=abs))}")

    # --- Iterative derivative convergence ---
    print("\n" + "-" * 70)
    print("ITERATIVE DIFFERENTIATION")
    print("-" * 70)

    all_roots = direct_polynomial_derivative_chain(z_vertices)

    # --- Analysis ---
    analyze_convergence(all_roots, cell_centroid)

    # Also check against the generator point
    generator = all_points[cell_index][0] + 1j * all_points[cell_index][1]
    final_point = np.mean(all_roots[-1])
    print(f"\nGenerator point:  {generator:.6f}")
    print(f"Distance final->generator: {abs(final_point - generator):.6f}")

    # --- Visualization ---
    print("\nGenerating plots...")

    # Plot 1: Main convergence visualization
    plot_derivative_convergence(all_roots, cell_centroid, vertices,
                                title=f"Cell {cell_index}, {len(vertices)} vertices")

    # Plot 2: Companion matrix evolution
    plot_companion_matrix_evolution(all_roots)

    # Plot 3: Eigenvalue spectrum
    plot_eigenvalue_spectrum(all_roots)

    # --- Additional experiment: multiple cells ---
    print("\n" + "=" * 70)
    print("MULTI-CELL COMPARISON")
    print("=" * 70)

    results = []
    for ci in range(min(n_points, 10)):
        verts, _ = get_voronoi_cell_vertices(all_points, ci)
        if verts is None or len(verts) < 3:
            continue

        z_v = vertices_to_complex(verts)
        centroid = np.mean(z_v)
        gen_pt = all_points[ci][0] + 1j * all_points[ci][1]

        roots_chain = direct_polynomial_derivative_chain(z_v)
        final = np.mean(roots_chain[-1])

        results.append({
            'cell': ci,
            'n_vertices': len(verts),
            'centroid': centroid,
            'generator': gen_pt,
            'final_point': final,
            'dist_to_centroid': abs(final - centroid),
            'dist_to_generator': abs(final - gen_pt)
        })

        print(f"Cell {ci:2d}: {len(verts)} verts | "
              f"final={final:.4f} | "
              f"d(centroid)={abs(final - centroid):.4f} | "
              f"d(generator)={abs(final - gen_pt):.4f}")

    # Summary
    if results:
        avg_dist_centroid = np.mean([r['dist_to_centroid'] for r in results])
        avg_dist_generator = np.mean([r['dist_to_generator'] for r in results])
        print(f"\nAverage distance to centroid:  {avg_dist_centroid:.6f}")
        print(f"Average distance to generator: {avg_dist_generator:.6f}")
        print(f"\nThe derivative chain converges closer to: "
              f"{'CENTROID' if avg_dist_centroid < avg_dist_generator else 'GENERATOR'}")

    # --- Bonus: Animated-style step-by-step for the chosen cell ---
    print("\n" + "=" * 70)
    print(f"STEP-BY-STEP FOR CELL {cell_index}")
    print("=" * 70)

    print(f"\nOriginal polynomial degree: {len(z_vertices)}")
    print(f"Vertices (= eigenvalues of C_0):")

    for step, roots in enumerate(all_roots):
        print(f"\n--- Derivative order {step} (degree {len(roots)}) ---")
        C = build_companion_matrix(roots) if len(roots) > 0 else None

        if C is not None:
            print(f"Companion matrix C_{step}:")
            for row in C:
                print("  [" + "  ".join(f"{x.real:8.4f}+{x.imag:8.4f}j" for x in row) + "]")

            eigenvals = np.linalg.eigvals(C)
            print(f"Eigenvalues: {eigenvals}")
            print(f"Centroid: {np.mean(eigenvals):.6f}")
            print(f"Spread: {np.std(eigenvals):.6f}")


if __name__ == "__main__":
    main()