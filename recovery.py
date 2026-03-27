"""
Voronoi Tessellation via Global Tessellation Matrix
====================================================

Spectral encoding of Voronoi tessellations using companion matrices:
  - Each Voronoi cell V_i with vertices {v_{i1},...,v_{ik_i}} (complex numbers)
    is encoded by the companion matrix C_i of P_i(z) = ∏(z - v_{ij})
  - Spec(C_i) = {v_{i1}, ..., v_{ik_i}}
  - Global tessellation matrix: C = block_diag(C_1, C_2, ..., C_m)
  - Spec(C) = ∪ Spec(C_i) = all Voronoi vertices
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.linalg import block_diag
from matplotlib.patches import Polygon
import matplotlib.cm as cm


# ----------------------------------------------------------------
# Utility: complex ↔ 2D coordinate conversion
# ----------------------------------------------------------------

def complex_to_2d(z):
    """Convert array of complex numbers to Nx2 real array."""
    z = np.asarray(z)
    return np.column_stack([z.real, z.imag])


def coords_to_complex(coords):
    """Convert Nx2 real coordinates to complex numbers."""
    coords = np.asarray(coords)
    return coords[:, 0] + 1j * coords[:, 1]


# ----------------------------------------------------------------
# Companion matrix construction
# ----------------------------------------------------------------

def build_companion_matrix(vertices_complex):
    """
    Build the companion matrix for a Voronoi cell.

    Given vertices v_1,...,v_n (complex), form:
        P(z) = (z-v_1)...(z-v_n) = z^n + a_{n-1}z^{n-1} + ... + a_0

    Companion matrix (matching the document's convention):
        C = | 0  0  ...  0  -a_0     |
            | 1  0  ...  0  -a_1     |
            | 0  1  ...  0  -a_2     |
            | :  :  ...  :   :       |
            | 0  0  ...  1  -a_{n-1} |

    Eigenvalues of C are exactly {v_1, ..., v_n}.
    """
    vertices_complex = np.asarray(vertices_complex)
    n = len(vertices_complex)

    # np.poly returns [1, a_{n-1}, a_{n-2}, ..., a_1, a_0]
    coeffs = np.poly(vertices_complex)

    C = np.zeros((n, n), dtype=complex)

    # Sub-diagonal ones
    for i in range(1, n):
        C[i, i - 1] = 1.0

    # Last column: [-a_0, -a_1, ..., -a_{n-1}]
    for k in range(n):
        C[k, n - 1] = -coeffs[n - k]

    return C


def build_global_tessellation_matrix(cells_vertices_list):
    """
    Build C = block_diag(C_1, ..., C_m).

    Parameters
    ----------
    cells_vertices_list : list of 1-D complex arrays
        Each element holds the complex vertices of one cell.

    Returns
    -------
    C : ndarray  – the global tessellation matrix
    block_sizes : list of int – number of vertices per cell
    """
    companions = [build_companion_matrix(v) for v in cells_vertices_list]
    block_sizes = [len(v) for v in cells_vertices_list]
    C = block_diag(*companions)
    return C, block_sizes


# ----------------------------------------------------------------
# Recovery from tessellation matrix
# ----------------------------------------------------------------

def detect_block_sizes(C, tol=1e-10):
    """
    Auto-detect block sizes in a block-diagonal matrix via
    connected-component analysis on the non-zero pattern.
    """
    n = C.shape[0]
    visited = [False] * n
    sizes = []

    for start in range(n):
        if not visited[start]:
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                for j in range(n):
                    if not visited[j] and (
                        abs(C[node, j]) > tol or abs(C[j, node]) > tol
                    ):
                        stack.append(j)
            sizes.append(len(component))
    return sizes


def extract_cells_from_tessellation_matrix(C, block_sizes=None):
    """
    Extract Voronoi-cell vertices as eigenvalues of each diagonal block.

    Parameters
    ----------
    C : ndarray – global tessellation matrix
    block_sizes : list of int or None (auto-detected)

    Returns
    -------
    cells : list of 1-D complex arrays (one per cell)
    """
    if block_sizes is None:
        block_sizes = detect_block_sizes(C)

    cells = []
    offset = 0
    for sz in block_sizes:
        C_i = C[offset : offset + sz, offset : offset + sz]
        cells.append(np.linalg.eigvals(C_i))
        offset += sz
    return cells


# ----------------------------------------------------------------
# Polygon vertex ordering
# ----------------------------------------------------------------

def order_polygon_vertices(pts_2d):
    """Order 2D points counter-clockwise around their centroid."""
    c = pts_2d.mean(axis=0)
    angles = np.arctan2(pts_2d[:, 1] - c[1], pts_2d[:, 0] - c[0])
    return pts_2d[np.argsort(angles)]


# ----------------------------------------------------------------
# Visualisation helpers
# ----------------------------------------------------------------

def print_matrix_info(C, block_sizes):
    """Print a short summary of the tessellation matrix."""
    nnz = np.count_nonzero(np.abs(C) > 1e-12)
    print(f"  Shape             : {C.shape}")
    print(f"  Cells             : {len(block_sizes)}")
    print(f"  Vertices per cell : {block_sizes}")
    print(f"  Total vertex slots: {sum(block_sizes)}")
    print(f"  Non-zeros         : {nnz}/{C.size}  "
          f"({100 * nnz / C.size:.1f}%)")


def plot_voronoi_from_matrix(C, block_sizes=None, generators=None,
                              title="Voronoi from Global Tessellation Matrix"):
    """
    Recover Voronoi cells from C and draw them as filled polygons.
    """
    cells = extract_cells_from_tessellation_matrix(C, block_sizes)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = cm.tab20(np.linspace(0, 1, max(len(cells), 1)))

    all_pts = []
    for i, eigs in enumerate(cells):
        v2d = complex_to_2d(eigs)
        all_pts.append(v2d)
        ordered = order_polygon_vertices(v2d)

        poly = Polygon(ordered, closed=True,
                       facecolor=palette[i % len(palette)],
                       edgecolor='black', linewidth=2, alpha=0.45)
        ax.add_patch(poly)
        ax.scatter(v2d[:, 0], v2d[:, 1], c='black', s=35, zorder=5)

        cx, cy = v2d.mean(axis=0)
        ax.text(cx, cy, f'$V_{{{i+1}}}$', fontsize=13,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.75))

    if generators is not None:
        g = np.asarray(generators)
        ax.scatter(g[:, 0], g[:, 1], c='red', s=120, marker='*',
                   zorder=10, edgecolors='darkred', linewidths=0.5,
                   label='Generator points')
        ax.legend(fontsize=12)

    all_pts = np.vstack(all_pts)
    
    ax.set_xlim(all_pts[:, 0].min() - mx, all_pts[:, 0].max() + mx)
    ax.set_ylim(all_pts[:, 1].min() - my, all_pts[:, 1].max() + my)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Re(z)', fontsize=12)
    ax.set_ylabel('Im(z)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return cells


# ================================================================
#  DEMONSTRATIONS
# ================================================================

if __name__ == "__main__":

    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    # ============================================================
    # Demo 1  –  scipy Voronoi → C → spectral recovery
    # ============================================================
    print("=" * 70)
    print(" DEMO 1: Voronoi → Tessellation Matrix → Eigenvalue Recovery")
    print("=" * 70)

    np.random.seed(42)
    points = np.random.rand(20, 2) * 10
    vor = Voronoi(points)

    # Keep only bounded (finite) cells
    cells_complex, finite_gens = [], []
    for idx, ri in enumerate(vor.point_region):
        region = vor.regions[ri]
        if len(region) > 0 and -1 not in region:
            cells_complex.append(coords_to_complex(vor.vertices[region]))
            finite_gens.append(points[idx])
    finite_gens = np.array(finite_gens)

    C, bsizes = build_global_tessellation_matrix(cells_complex)

    print(f"\nGenerator points      : {len(points)}")
    print(f"Bounded cells found   : {len(cells_complex)}")
    print_matrix_info(C, bsizes)

    # Verify round-trip
    recovered = extract_cells_from_tessellation_matrix(C, bsizes)
    print("\nReconstruction errors:")
    for i, (o, r) in enumerate(zip(cells_complex, recovered)):
        o_s = o[np.lexsort((o.imag, o.real))]
        r_s = r[np.lexsort((r.imag, r.real))]
        err = np.max(np.abs(o_s - r_s))
        print(f"  Cell {i+1:2d}  ({len(o)} verts)  err = {err:.2e}")

    # Side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    voronoi_plot_2d(vor, ax=ax1, show_vertices=True,
                    line_colors='black', line_width=2, point_size=8)
    ax1.set_title("Original scipy.spatial.Voronoi", fontsize=14)
    ax1.set_xlim(-1, 11); ax1.set_ylim(-1, 11)
    ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)

    palette = cm.tab20(np.linspace(0, 1, len(recovered)))
    for i, eigs in enumerate(recovered):
        v = complex_to_2d(eigs)
        vo = order_polygon_vertices(v)
        ax2.add_patch(Polygon(vo, closed=True,
                              facecolor=palette[i], edgecolor='black',
                              linewidth=2, alpha=0.4))
        ax2.scatter(v[:, 0], v[:, 1], c='black', s=25, zorder=5)
    ax2.scatter(finite_gens[:, 0], finite_gens[:, 1],
                c='red', s=100, marker='*', zorder=10, label='Generators')
    ax2.set_title("Recovered from Spec(C)", fontsize=14)
    ax2.set_xlim(-1, 11); ax2.set_ylim(-1, 11)
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3); ax2.legend()
    plt.tight_layout(); plt.show()



    # ============================================================
    # Demo 3  –  Given ONLY the matrix, recover the diagram
    # ============================================================
    print("\n" + "=" * 70)
    print(" DEMO 3: Matrix-only recovery (block sizes auto-detected)")
    print("=" * 70)

    # Build a matrix the user might provide
    hex_v  = np.exp(1j * np.linspace(0, 2*np.pi, 7)[:-1]) + (3+3j)
    pent_v = np.exp(1j * np.linspace(0, 2*np.pi, 6)[:-1]) + (6+3j)
    tri_v  = np.exp(1j * np.linspace(0, 2*np.pi, 4)[:-1]) + (4.5+5.5j)

    C_only, _ = build_global_tessellation_matrix([hex_v, pent_v, tri_v])

    # Pretend we only have C_only
    auto_bs = detect_block_sizes(C_only)
    print(f"  Auto-detected block sizes: {auto_bs}")
    rec = extract_cells_from_tessellation_matrix(C_only)
    for i, c in enumerate(rec):
        print(f"  Cell {i+1}: {len(c)} vertices recovered")

    plot_voronoi_from_matrix(
        C_only,
        title="Demo 3: Recovery from matrix alone\n"
              "(hexagon + pentagon + triangle)")

    # ============================================================
    # Demo 4  –  Spectrum visualisation  Spec(C) = ∪ Spec(C_i)
    # ============================================================
    print("\n" + "=" * 70)
    print(" DEMO 4: Spectrum Visualisation")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(9, 8))
    offset = 0
    for i, sz in enumerate(bsizes):
        blk = C[offset:offset+sz, offset:offset+sz]
        eigs = np.linalg.eigvals(blk)
        ax.scatter(eigs.real, eigs.imag, s=60,
                   edgecolors='black', linewidths=0.5,
                   label=f'Cell {i+1}', zorder=5)
        offset += sz

    ax.set_title(r"$\mathrm{Spec}(\mathcal{C}) = "
                 r"\bigcup_i \mathrm{Spec}(C_i)$"
                 "\n(each colour = one Voronoi cell)", fontsize=14)
    ax.set_xlabel("Re(z)", fontsize=12)
    ax.set_ylabel("Im(z)", fontsize=12)
    ax.legend(fontsize=8, ncol=3, loc='best')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    print("\nAll demos complete.")