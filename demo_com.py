import numpy as np
import matplotlib.pyplot as plt

def demo(vertices):

    # compute centroid
    centroid = np.mean(vertices)

    # construct polynomial
    P = np.poly(vertices)

    current_poly = P

    roots_history = []

    while len(current_poly) > 2:
        current_poly = np.polyder(current_poly)
        roots = np.roots(current_poly)
        roots_history.append(roots)

    final_root = -current_poly[1]/current_poly[0]

    print("Centroid:", centroid)
    print("Final root:", final_root)
    print("Difference:", abs(final_root-centroid))

    return roots_history

def plot_contraction(vertices):

    centroid = np.mean(vertices)
    P = np.poly(vertices)

    current_poly = P

    plt.scatter(vertices.real, vertices.imag, label="Original roots")

    while len(current_poly) > 2:
        current_poly = np.polyder(current_poly)
        roots = np.roots(current_poly)
        plt.scatter(roots.real, roots.imag)

    plt.scatter(centroid.real, centroid.imag,
                color='red', label="Centroid")

    plt.legend()
    plt.axis('equal')
    plt.show()

vertices = np.array([
    1 + 2j,
    3 + 1j,
    2 - 2j,
    -1 - 1j,
    -2 + 1j
])

demo(vertices)
plot_contraction(vertices)