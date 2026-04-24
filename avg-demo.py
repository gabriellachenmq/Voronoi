import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree

# ------------------------------
# 1. Fixed boundary points
# ------------------------------
def create_boundary_points(xmin, xmax, ymin, ymax, n_per_edge=5):
    """
    Create fixed points along the rectangle boundary.
    n_per_edge: number of points per edge (including corners).
    """
    x_vals = np.linspace(xmin, xmax, n_per_edge)
    y_vals = np.linspace(ymin, ymax, n_per_edge)

    bottom = np.array([[x, ymin] for x in x_vals])
    top    = np.array([[x, ymax] for x in x_vals])
    left   = np.array([[xmin, y] for y in y_vals[1:-1]])   # exclude corners already added
    right  = np.array([[xmax, y] for y in y_vals[1:-1]])

    boundary = np.vstack([bottom, top, left, right])
    return boundary

# ------------------------------
# 2. Main iteration
# ------------------------------
def voronoi_vertex_mean(points, bounding_box=None, static_mask=None):
    """
    For each point that is not static (static_mask=False), compute the mean of its
    Voronoi cell vertices. Static points are fixed and never updated.

    Parameters:
    -----------
    points : (N,2) array
        All points (static + moving)
    bounding_box : (xmin,xmax,ymin,ymax) or None
        Not used directly, but assumed that cells are bounded because of static points.
    static_mask : bool array, shape (N,)
        True for points that should not be moved.

    Returns:
    --------
    new_points : (N,2) array
        Updated positions (static points unchanged)
    """
    vor = Voronoi(points)
    new_points = points.copy()

    # For each moving point (static_mask == False)
    for i, is_static in enumerate(static_mask):
        if is_static:
            continue

        # Voronoi region index for point i
        region_idx = vor.point_region[i]
        vertices_indices = vor.regions[region_idx]

        # If region contains -1, the cell is unbounded – should not happen with our boundary setup
        if -1 in vertices_indices:
            print(f"Warning: point {i} has unbounded cell – ignoring update")
            continue

        # Get actual coordinates of vertices
        verts = vor.vertices[vertices_indices]
        if len(verts) == 0:
            continue

        # Mean of vertices
        centroid = np.mean(verts, axis=0)
        new_points[i] = centroid

    return new_points

# ------------------------------
# 3. Demo
# ------------------------------
def run_demo(n_moving=20, n_iter=30, tol=1e-4, plot_every=5):
    # Domain
    xmin, xmax = 0, 10
    ymin, ymax = 0, 10

    # Create fixed boundary points
    boundary = create_boundary_points(xmin, xmax, ymin, ymax, n_per_edge=8)
    n_boundary = len(boundary)

    # Random interior points (moving)
    margin = 0.5
    moving = np.random.uniform(
        low=[xmin+margin, ymin+margin],
        high=[xmax-margin, ymax-margin],
        size=(n_moving, 2)
    )

    # Combine all points
    points = np.vstack([boundary, moving])
    static_mask = np.array([True]*n_boundary + [False]*n_moving)

    # Store history for later plotting
    history = [points[n_boundary:].copy()]

    # Iteration
    for it in range(n_iter):
        points_new = voronoi_vertex_mean(points, static_mask=static_mask)
        # Compute movement of moving points only
        moving_old = points[n_boundary:]
        moving_new = points_new[n_boundary:]
        max_move = np.max(np.linalg.norm(moving_new - moving_old, axis=1))

        points = points_new
        history.append(moving_new.copy())

        print(f"Iter {it+1:2d}, max movement = {max_move:.6f}")
        if max_move < tol:
            print("Converged.")
            break

        # Optional: plot every few iterations
        #if (it+1) % plot_every == 0 or it == 0:
            #plot_voronoi(points, static_mask, title=f"Iteration {it+1}")

    # Final plot
    plot_voronoi(points, static_mask, title="Final configuration")
    # Plot movement trajectories
    plot_trajectories(history, xmin, xmax, ymin, ymax)

def plot_voronoi(points, static_mask, title="Voronoi diagram"):
    """Plot Voronoi diagram, static points as squares, moving points as dots."""
    vor = Voronoi(points)
    fig, ax = plt.subplots(figsize=(8, 8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray',
                    line_width=1, line_alpha=0.6, point_size=2)
    # Static points
    static_pts = points[static_mask]
    ax.scatter(static_pts[:,0], static_pts[:,1], c='red', marker='s', s=60,
               label='Fixed boundary', edgecolors='k')
    # Moving points
    moving_pts = points[~static_mask]
    ax.scatter(moving_pts[:,0], moving_pts[:,1], c='blue', marker='o', s=60,
               label='Moving generators', edgecolors='k')
    ax.set_xlim(points[:,0].min()-0.5, points[:,0].max()+0.5)
    ax.set_ylim(points[:,1].min()-0.5, points[:,1].max()+0.5)
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

def plot_trajectories(history, xmin, xmax, ymin, ymax):
    """Plot the path of each moving point over iterations."""
    history = np.array(history)  # shape (n_iter+1, n_moving, 2)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(history.shape[1]):
        ax.plot(history[:, i, 0], history[:, i, 1], 'o-', markersize=3, linewidth=0.8, alpha=0.7)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Trajectories of moving generators")
    ax.set_aspect('equal')
    plt.show()

# ------------------------------
# Run the demo
# ------------------------------
if __name__ == "__main__":
    run_demo(n_moving=40, n_iter=100, tol=1e-5, plot_every=8)