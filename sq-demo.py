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

# --------------------------------------------------
# Define shared edge
# --------------------------------------------------

v_shared1 = 1 + 0.2j
v_shared2 = 1.2 + 1.1j

# --------------------------------------------------
# Define cells (sharing the edge)
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
# Plot
# --------------------------------------------------

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for step in range(5):
    ax = axes[step]

    for idx, cell in enumerate(cells):
        coeffs = poly_from_roots(cell)

        # differentiate step-by-step
        for _ in range(step):
            if len(coeffs) > 1:
                coeffs = differentiate_poly(coeffs)

        roots_k = roots_from_poly(coeffs)
        roots_k = order_polygon(roots_k)

        x = roots_k.real
        y = roots_k.imag

        if len(roots_k) > 2:
            ax.plot(np.append(x, x[0]), np.append(y, y[0]),
                    'o-', label=f'Cell {idx+1}')
        else:
            ax.scatter(x, y, s=80, label=f'Cell {idx+1}')

    ax.set_title(f"Step {step}")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True)

axes[0].legend()
plt.suptitle("Differentiation Flow with Shared Edge (Quad + Pentagon)")
plt.show()