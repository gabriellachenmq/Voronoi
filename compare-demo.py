import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def poly_from_roots(roots):
    return np.poly(roots)

def differentiate_poly(coeffs):
    n = len(coeffs) - 1
    return np.array([coeffs[i] * (n - i) for i in range(n)])

def roots_from_poly(coeffs):
    return np.roots(coeffs)

def order_polygon(roots):
    roots = np.array(roots)
    center = np.mean(roots)
    angles = np.angle(roots - center)
    order = np.argsort(angles)
    return roots[order]

def differentiate_until_point(roots):
    """Apply differentiation repeatedly until one root remains"""
    coeffs = poly_from_roots(roots)

    while len(coeffs) > 2:  # degree > 1
        coeffs = differentiate_poly(coeffs)

    final_root = roots_from_poly(coeffs)[0]
    return final_root

# --------------------------------------------------
# Define shared edge
# --------------------------------------------------

v_shared1 = 1 + 0.2j
v_shared2 = 1.2 + 1.1j

# --------------------------------------------------
# Define cells
# --------------------------------------------------

quad = np.array([
    0 + 0j,
    1 + 0j,
    v_shared1,
    v_shared2
])

pent = np.array([
    v_shared1,
    v_shared2,
    2.5 + 0.2j,
    2.8 + 1.5j,
    1.8 + 1.8j
])

cells = [quad, pent]

# --------------------------------------------------
# Compute comparison
# --------------------------------------------------

print("Comparison: Differentiation vs Direct Average\n")

for i, cell in enumerate(cells):
    # differentiation result
    diff_centroid = differentiate_until_point(cell)

    # direct average (Tz)
    direct_centroid = np.mean(cell)

    print(f"Cell {i+1}:")
    print(f"  Differentiation result: {diff_centroid}")
    print(f"  Direct average:         {direct_centroid}")
    print(f"  Difference:             {abs(diff_centroid - direct_centroid)}\n")

# --------------------------------------------------
# Visualization
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 6))

for idx, cell in enumerate(cells):
    ordered = order_polygon(cell)

    x = ordered.real
    y = ordered.imag

    # plot original polygon
    ax.plot(np.append(x, x[0]), np.append(y, y[0]), 'o-', label=f'Cell {idx+1}')

    # compute centroids
    diff_centroid = differentiate_until_point(cell)
    direct_centroid = np.mean(cell)

    # plot differentiation centroid
    ax.scatter(diff_centroid.real, diff_centroid.imag,
               marker='x', s=120, label=f'Diff centroid {idx+1}')

    # plot direct centroid
    ax.scatter(direct_centroid.real, direct_centroid.imag,
               marker='*', s=120, label=f'Avg centroid {idx+1}')

ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 2.5)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title("Differentiation vs Direct Averaging")

plt.show()