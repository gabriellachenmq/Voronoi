import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_polygon(points, color, label=None):
    if len(points) < 3:
        plt.scatter(points.real, points.imag, color=color, label=label)
        return

    pts = np.column_stack((points.real, points.imag))
    hull = ConvexHull(pts)
    hull_points = pts[hull.vertices]

    # close the polygon
    hull_points = np.vstack([hull_points, hull_points[0]])

    plt.plot(hull_points[:,0], hull_points[:,1], color=color)
    plt.scatter(points.real, points.imag, color=color, label=label)


def voronoi_derivative_demo(vertices):

    centroid = np.mean(vertices)

    P = np.poly(vertices)
    current_poly = P

    colors = ["blue","green","orange","purple","brown","gray"]

    plt.figure(figsize=(6,6))

    # original polygon
    plot_polygon(vertices, colors[0], "P roots")

    step = 1

    while len(current_poly) > 2:
        current_poly = np.polyder(current_poly)
        roots = np.roots(current_poly)

        plot_polygon(roots, colors[min(step,len(colors)-1)], f"P^{step} roots")

        step += 1

    # final root of linear polynomial
    final_root = -current_poly[1] / current_poly[0]

    # numerical diagnostics
    diff = abs(final_root - centroid)

    print("\n--- Numerical Results ---")
    print("Centroid:", centroid)
    print("Final root after repeated differentiation:", final_root)
    print("Absolute difference:", diff)
    print("Real difference:", final_root.real - centroid.real)
    print("Imag difference:", final_root.imag - centroid.imag)

    plt.scatter(centroid.real, centroid.imag,
                color="red", s=100, label="centroid")

    plt.scatter(final_root.real, final_root.imag,
                color="black", s=60, label="final root")

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)

    plt.legend()
    plt.axis("equal")
    plt.title("Gauss–Lucas Polygon Contraction")
    plt.show()



vertices = np.array([
    1+2j,
    3+1j,
    2-2j,
    -1-1j,
    -2+1j
])

voronoi_derivative_demo(vertices)