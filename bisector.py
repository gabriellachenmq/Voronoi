import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiPoint
import copy
import random
from sklearn.cluster import KMeans


class VoronoiGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("CVT Generator with Radiant Algorithm")

        self.points = []
        self.original_points = []
        self.vor = None
        self.original_vor = None
        self.centroids = None
        self.iteration_count = 0
        self.radiant_iterations = 0
        self.auto_running = False
        self.auto_previous_points = None
        self.frozen_points = set()

        self.fixed_point_mode = False
        self.fixed_point_index = None

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.width = 1200
        self.height = 1200
        self.click_canvas = tk.Canvas(self.main_frame, bg='white', width=self.width, height=self.width)
        self.click_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        self.random_button = tk.Button(self.button_frame, text="Generate Random Points",
                                       command=self.generate_random_points)
        self.random_button.pack(fill=tk.X, pady=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear Points", command=self.clear_points)
        self.clear_button.pack(fill=tk.X, pady=5)

        self.generate_button = tk.Button(self.button_frame, text="Generate Voronoi", command=self.generate_voronoi)
        self.generate_button.pack(fill=tk.X, pady=5)

        self.lloyd_button = tk.Button(self.button_frame, text="Lloyd's Algorithm", command=self.apply_lloyd)
        self.lloyd_button.pack(fill=tk.X, pady=5)

        self.auto_lloyd_button = tk.Button(self.button_frame, text="Auto CVT", command=self.auto_lloyd)
        self.auto_lloyd_button.pack(fill=tk.X, pady=5)

        self.radiant_button = tk.Button(self.button_frame, text="Run Radiant Algorithm",
                                        command=self.apply_radiant_algorithm)
        self.radiant_button.pack(fill=tk.X, pady=5)

        self.show_original_button = tk.Button(self.button_frame, text="Show Original Points",
                                              command=self.show_original_points)
        self.show_original_button.pack(fill=tk.X, pady=5)

        self.info_label = tk.Label(self.button_frame, text="Iterations: 0")
        self.info_label.pack(fill=tk.X, pady=5)

        self.progressive_radiant_central_btn = tk.Button(
            self.button_frame,
            text="Central Outward Progressive Radiant",
            command=self.progressive_radiant_central_outward
        )
        self.progressive_radiant_central_btn.pack(fill=tk.X, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(self.width / 100, self.height / 100))
        self.ax.set_aspect('equal')
        self.canvas = None

        self.bounds = [0, self.width, 0, self.height]
        self.width = self.bounds[1] - self.bounds[0]
        self.height = self.bounds[3] - self.bounds[2]

    # =====================================================================
    #  NEW: Perpendicular Bisector Intersection (replaces polygon centroid)
    # =====================================================================

    def find_perpendicular_bisector_intersection(self, polygon):
        """
        For a given polygon, compute the perpendicular bisector of each side,
        then find the best-fit intersection point of all bisectors via least squares.

        The perpendicular bisector of edge P1->P2 is the locus of points equidistant
        from P1 and P2. Its equation is:
            (P2 - P1) · (X - midpoint) = 0
        i.e.  dx*x + dy*y = dx*mx + dy*my

        For a cyclic polygon (inscribed in a circle), all bisectors meet at one point
        (the circumcenter). For a general polygon, we solve via least squares.

        Returns: np.array([x, y]) — the best-fit intersection point.
        """
        coords = list(polygon.exterior.coords)
        # Remove the closing vertex (duplicate of first)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]

        n = len(coords)
        if n < 2:
            # Degenerate — fall back to centroid
            return np.array([polygon.centroid.x, polygon.centroid.y])

        # Build system: for each edge, one equation  dx*x + dy*y = c
        A_mat = []
        b_vec = []

        for i in range(n):
            p1 = np.array(coords[i], dtype=float)
            p2 = np.array(coords[(i + 1) % n], dtype=float)

            midpoint = (p1 + p2) / 2.0
            d = p2 - p1  # edge direction vector

            edge_len = np.linalg.norm(d)
            if edge_len < 1e-12:
                continue  # skip degenerate zero-length edges

            # Equation: d[0]*x + d[1]*y = d[0]*mx + d[1]*my
            A_mat.append([d[0], d[1]])
            b_vec.append(d[0] * midpoint[0] + d[1] * midpoint[1])

        A_mat = np.array(A_mat)
        b_vec = np.array(b_vec)

        if len(A_mat) < 2:
            return np.array([polygon.centroid.x, polygon.centroid.y])

        try:
            result, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            return result  # [x, y]
        except np.linalg.LinAlgError:
            return np.array([polygon.centroid.x, polygon.centroid.y])

    def get_perpendicular_bisectors(self, polygon):
        """
        Return a list of bisector lines for visualization.
        Each bisector is represented as (midpoint, direction, length_hint)
        where the line passes through midpoint in the perpendicular direction.
        """
        coords = list(polygon.exterior.coords)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]

        bisectors = []
        n = len(coords)
        for i in range(n):
            p1 = np.array(coords[i], dtype=float)
            p2 = np.array(coords[(i + 1) % n], dtype=float)
            midpoint = (p1 + p2) / 2.0
            d = p2 - p1
            edge_len = np.linalg.norm(d)
            if edge_len < 1e-12:
                continue
            # Perpendicular direction to the edge (rotate 90°)
            perp = np.array([-d[1], d[0]])
            perp = perp / np.linalg.norm(perp)
            bisectors.append((midpoint, perp, edge_len))
        return bisectors

    def compute_bisector_residual(self, polygon):
        """
        Compute how far the bisectors are from meeting at a single point.
        Returns the RMS residual from the least-squares solve.
        Small residual ≈ polygon is nearly cyclic (inscribed in a circle).
        """
        coords = list(polygon.exterior.coords)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]

        n = len(coords)
        if n < 3:
            return float('inf')

        A_mat = []
        b_vec = []
        for i in range(n):
            p1 = np.array(coords[i], dtype=float)
            p2 = np.array(coords[(i + 1) % n], dtype=float)
            midpoint = (p1 + p2) / 2.0
            d = p2 - p1
            if np.linalg.norm(d) < 1e-12:
                continue
            A_mat.append([d[0], d[1]])
            b_vec.append(d[0] * midpoint[0] + d[1] * midpoint[1])

        A_mat = np.array(A_mat)
        b_vec = np.array(b_vec)

        if len(A_mat) < 2:
            return float('inf')

        try:
            result, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            # Compute residuals manually if not returned
            predicted = A_mat @ result
            res = np.sqrt(np.mean((predicted - b_vec) ** 2))
            return res
        except np.linalg.LinAlgError:
            return float('inf')

    # =====================================================================
    #  Existing methods (unchanged unless noted with MODIFIED comment)
    # =====================================================================

    def generate_random_points(self):
        seed = simpledialog.askinteger("Random Seed", "Enter random seed:", parent=self.root)
        num_points = simpledialog.askinteger("Number of Points", "Points to generate:",
                                             parent=self.root, minvalue=2, initialvalue=20)
        if num_points is None:
            return
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.clear_points()

        initial_points = np.random.rand(num_points * 10, 2)
        initial_points[:, 0] = initial_points[:, 0] * (self.width - 100) + 50
        initial_points[:, 1] = initial_points[:, 1] * (self.height - 100) + 50

        kmeans = KMeans(n_clusters=num_points, random_state=seed if seed else None)
        kmeans.fit(initial_points)
        self.points = kmeans.cluster_centers_.tolist()
        self.original_points = copy.deepcopy(self.points)

        self.click_canvas.delete("all")
        for x, y in self.points:
            canvas_y = self.height - y
            self.click_canvas.create_oval(x - 3, canvas_y - 3, x + 3, canvas_y + 3, fill='red')

    def clear_points(self):
        self.points = []
        self.original_points = []
        self.vor = None
        self.centroids = None
        self.iteration_count = 0
        self.auto_running = False
        self.radiant_running = False
        self.fixed_point_index = None
        self.info_label.config(text="Iterations: 0")
        self.click_canvas.delete("all")
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.ax.clear()
        # MODIFIED: clear bisector tracking data
        if hasattr(self, 'previous_bisector_centers'):
            self.previous_bisector_centers = {}
        if hasattr(self, 'previous_centroids'):
            self.previous_centroids = {}

    def add_periodic_ghosts(self, points):
        points = np.array(points)
        tile_shifts = [
            (0, 0),
            (self.width, 0), (-self.width, 0),
            (0, self.height), (0, -self.height),
            (self.width, self.height), (self.width, -self.height),
            (-self.width, self.height), (-self.width, -self.height)
        ]
        all_points = []
        for dx, dy in tile_shifts:
            shifted = points + np.array([dx, dy])
            all_points.append(shifted)
        return np.vstack(all_points)

    def generate_voronoi(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "Need at least 2 points")
            return
        self.points = np.array(self.points)
        self.vor = Voronoi(self.add_periodic_ghosts(self.points))
        self.iteration_count = 0
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

    def plot_voronoi(self):
        self.ax.clear()
        bounding_box = Polygon([
            (self.bounds[0], self.bounds[2]),
            (self.bounds[1], self.bounds[2]),
            (self.bounds[1], self.bounds[3]),
            (self.bounds[0], self.bounds[3])
        ])
        for i, region_index in enumerate(self.vor.point_region[:len(self.points)]):
            region = self.vor.regions[region_index]
            if -1 in region or len(region) == 0:
                continue
            polygon = Polygon(self.vor.vertices[region])
            clipped = polygon.intersection(bounding_box)
            if clipped.is_empty:
                continue
            if isinstance(clipped, Polygon):
                self.ax.fill(*clipped.exterior.xy, alpha=0.4)
            elif isinstance(clipped, MultiPolygon):
                for poly in clipped.geoms:
                    self.ax.fill(*poly.exterior.xy, alpha=0.4)

        for i, point in enumerate(self.points):
            self.ax.plot(point[0], point[1], 'o', color='red', markersize=5)

        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[2], self.bounds[3])
        self.ax.set_title(f"Voronoi Diagram (Iteration: {self.iteration_count})")

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.click_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def apply_lloyd(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Generate Voronoi first")
            return
        self.centroids = self.calculate_centroids()
        centroids_wrapped = []
        for x, y in self.centroids:
            centroids_wrapped.append((x % self.width, y % self.height))
        self.centroids = np.array(centroids_wrapped)
        self.vor = Voronoi(self.add_periodic_ghosts(self.centroids))
        self.points = self.centroids.copy()
        self.iteration_count += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

    def auto_lloyd(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Generate Voronoi first")
            return
        if self.auto_running:
            self.auto_running = False
            self.auto_lloyd_button.config(text="Auto CVT")
            return
        self.auto_running = True
        self.auto_lloyd_button.config(text="Stop CVT")
        self.auto_tolerance = 1e-3
        self.run_auto_lloyd()

    def run_auto_lloyd(self):
        if not self.auto_running:
            return
        current_points = np.array(self.points)
        for _ in range(10):
            self.apply_lloyd()
        if self.auto_previous_points is not None:
            movement = np.linalg.norm(self.auto_previous_points - self.points)
            if movement < self.auto_tolerance or self.iteration_count == 800:
                self.auto_running = False
                self.auto_lloyd_button.config(text="Auto CVT")
                self.select_most_central_point()
                return
        self.auto_previous_points = current_points
        self.root.after(100, self.run_auto_lloyd)

    def select_most_central_point(self):
        if len(self.points) == 0:
            return
        center_x, center_y = self.width / 2, self.height / 2
        min_dist = float('inf')
        central_idx = 0
        for i, (x, y) in enumerate(self.points):
            dist = (x - center_x) ** 2 + (y - center_y) ** 2
            if dist < min_dist:
                min_dist = dist
                central_idx = i
        self.fixed_point_index = central_idx
        self.plot_voronoi()
        self.ax.plot(self.points[central_idx][0], self.points[central_idx][1],
                     'o', color='blue', markersize=8, label='Central Point')
        self.ax.legend()
        self.canvas.draw()

    # =====================================================================
    #  MODIFIED: Convergence Detection — uses bisector intersection
    # =====================================================================

    def polygon_is_converged(self, point_index, poly, eps=2):
        """
        MODIFIED: Instead of tracking the centroid, we track the perpendicular
        bisector intersection point. If it barely moves between iterations,
        we consider the polygon converged.

        Additionally, we check the bisector residual — if it's small, the polygon
        is nearly cyclic (all bisectors nearly concurrent).
        """
        # Compute bisector intersection for this polygon
        new_center = self.find_perpendicular_bisector_intersection(poly)

        # Also compute how concurrent the bisectors are
        residual = self.compute_bisector_residual(poly)

        if not hasattr(self, "previous_bisector_centers"):
            self.previous_bisector_centers = {}

        if point_index not in self.previous_bisector_centers:
            self.previous_bisector_centers[point_index] = new_center
            return False

        old_center = self.previous_bisector_centers[point_index]
        dist = np.linalg.norm(new_center - old_center)
        self.previous_bisector_centers[point_index] = new_center

        # Converged if bisector intersection is stable AND bisectors are nearly concurrent
        return dist < eps and residual < eps

    def is_converged(self, level_polygons):
        """
        MODIFIED: Uses bisector-intersection-based convergence check.
        """
        for level, poly in level_polygons:
            neighbor_indices = self.get_k_level_neighbors(self.fixed_point_index, level)
            for point_index in neighbor_indices:
                if not self.polygon_is_converged(point_index, poly):
                    return False
        return True

    # =====================================================================
    #  Unchanged helper methods
    # =====================================================================

    def choose_new_center(self):
        all_indices = set(range(len(self.points)))
        candidates = list(all_indices - self.frozen_points)
        if not candidates:
            return None
        if not self.frozen_points:
            cx, cy = self.width / 2, self.height / 2
            distances = [(i, (self.points[i][0] - cx) ** 2 + (self.points[i][1] - cy) ** 2) for i in candidates]
            return max(distances, key=lambda x: x[1])[0]
        frozen_coords = np.array([self.points[i] for i in self.frozen_points])
        best_idx = None
        best_dist = -1
        for i in candidates:
            p = np.array(self.points[i])
            dist = np.min(np.linalg.norm(frozen_coords - p, axis=1))
            if dist > best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def run_radiant_algorithm(self):
        if self.fixed_point_index is None:
            messagebox.showwarning("Warning", "No central point. Run CVT first.")
            return
        if self.vor is None:
            messagebox.showwarning("Warning", "Generate Voronoi first.")
            return
        if self.radiant_running:
            self.radiant_running = False
            return
        self.radiant_running = True
        self.radiant_iterations = 0
        self.run_radiant_iteration()

    def run_radiant_iteration(self):
        if not self.radiant_running or self.radiant_iterations >= self.max_radiant_iterations:
            self.radiant_running = False
            return
        self.apply_radiant_algorithm()
        self.radiant_iterations += 1
        self.root.after(100, self.run_radiant_iteration)

    def is_boundary_point(self, point_idx):
        if point_idx >= len(self.points):
            return True
        region_index = self.vor.point_region[point_idx]
        if region_index == -1:
            return True
        region = self.vor.regions[region_index]
        if not region:
            return True
        for vertex_idx in region:
            if vertex_idx == -1:
                return True
            vertex = self.vor.vertices[vertex_idx]
            if (vertex[0] <= self.bounds[0] + 1e-6 or
                    vertex[0] >= self.bounds[1] - 1e-6 or
                    vertex[1] <= self.bounds[2] + 1e-6 or
                    vertex[1] >= self.bounds[3] - 1e-6):
                return True
        return False

    def calculate_max_levels(self, central_idx):
        max_levels = 0
        current_level = {central_idx}
        visited = set(current_level)
        boundary_found = False
        while not boundary_found:
            next_level = set()
            for point_idx in current_level:
                neighbors = self.get_voronoi_neighbors(point_idx)
                for neighbor_idx in neighbors:
                    if neighbor_idx not in visited:
                        if self.is_boundary_point(neighbor_idx):
                            boundary_found = True
                            break
                        next_level.add(neighbor_idx)
                        visited.add(neighbor_idx)
                if boundary_found:
                    break
            if not boundary_found and next_level:
                max_levels += 1
                current_level = next_level
            else:
                break
        return max_levels if max_levels > 0 else 1

    # =====================================================================
    #  MODIFIED: apply_radiant_algorithm — uses bisector intersection
    # =====================================================================

    def apply_radiant_algorithm(self):
        if self.fixed_point_index is None:
            messagebox.showwarning("Warning", "No central point. Run CVT first.")
            return
        if self.vor is None:
            messagebox.showwarning("Warning", "Generate Voronoi first.")
            return

        # --- Step 1: Check convergence ---
        level_polygons_short = self.get_neighborhood_polygons(self.fixed_point_index, 3)
        if level_polygons_short and self.is_converged(level_polygons_short):
            neighbor_ring = self.get_k_level_neighbors(self.fixed_point_index, 2)
            self.frozen_points.update(neighbor_ring)
            self.frozen_points.add(self.fixed_point_index)
            print(f"Freezing: {self.fixed_point_index} and neighbors {neighbor_ring}")
            new_center_idx = self.choose_new_center()
            if new_center_idx is None:
                messagebox.showinfo("Done", "Global convergence reached — no new center available.")
                return
            print(f"Switching center to point index {new_center_idx}")
            self.fixed_point_index = new_center_idx

        # --- Step 2: Standard Lloyd update ---
        centroids = self.calculate_centroids()
        for i, centroid in enumerate(centroids):
            if i in self.frozen_points:
                continue
            self.points[i] = (centroid[0] % self.width, centroid[1] % self.height)

        # --- MODIFIED Step 3: Use bisector intersection for central point ---
        level_polygons = self.get_neighborhood_polygons(self.fixed_point_index, 2)
        if level_polygons:
            # Compute bisector intersection for each level polygon
            bisector_centers = []
            for lvl, poly in level_polygons:
                bc = self.find_perpendicular_bisector_intersection(poly)
                bisector_centers.append(bc)

            # Average of bisector intersection points across levels
            bisector_avg = np.mean(bisector_centers, axis=0)

            # Blend with Lloyd centroid (50/50)
            lloyd_centroid = centroids[self.fixed_point_index]
            new_x = (0.5 * lloyd_centroid[0] + 0.5 * bisector_avg[0]) % self.width
            new_y = (0.5 * lloyd_centroid[1] + 0.5 * bisector_avg[1]) % self.height
            self.points[self.fixed_point_index] = (new_x, new_y)

        # --- Step 4: Update Voronoi ---
        self.vor = Voronoi(self.add_periodic_ghosts(self.points))
        self.iteration_count += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")

        # --- Step 5: Draw ---
        level_polygons_plot = self.get_neighborhood_polygons(self.fixed_point_index, 3)
        self.plot_radiant_voronoi(level_polygons_plot)

    # =====================================================================
    #  MODIFIED: plot_radiant_voronoi — draws bisectors + intersection pts
    # =====================================================================

    def plot_radiant_voronoi(self, level_polygons):
        self.ax.clear()

        bounding_box = Polygon([
            (self.bounds[0], self.bounds[2]),
            (self.bounds[1], self.bounds[2]),
            (self.bounds[1], self.bounds[3]),
            (self.bounds[0], self.bounds[3])
        ])

        for i, region_index in enumerate(self.vor.point_region[:len(self.points)]):
            region = self.vor.regions[region_index]
            if -1 in region or len(region) == 0:
                continue
            polygon = Polygon(self.vor.vertices[region])
            clipped = polygon.intersection(bounding_box)
            if clipped.is_empty:
                continue
            if isinstance(clipped, Polygon):
                self.ax.fill(*clipped.exterior.xy, alpha=0.4)
            elif isinstance(clipped, MultiPolygon):
                for poly in clipped.geoms:
                    self.ax.fill(*poly.exterior.xy, alpha=0.4)

        # Plot central point
        if self.fixed_point_index is not None:
            self.ax.plot(self.points[self.fixed_point_index][0],
                         self.points[self.fixed_point_index][1],
                         'o', color='blue', markersize=8, label='Central Point')

        # Color scheme for level polygons
        level_colors = ['green', 'orange', 'purple', 'cyan', 'magenta', 'brown']

        max_level = max(level for level, _ in level_polygons) if level_polygons else 0
        min_level = min(level for level, _ in level_polygons) if level_polygons else 0

        for level, poly in level_polygons:
            neighbor_indices = self.get_k_level_neighbors(self.fixed_point_index, level)
            color = level_colors[(level - 1) % len(level_colors)]

            if level == max_level and not any(self.is_boundary_point(i) for i in neighbor_indices):
                self.ax.plot(*poly.exterior.xy, color='black',
                             linewidth=2, linestyle='--',
                             label=f'Level {level} polygon')
            elif level == min_level:
                self.ax.plot(*poly.exterior.xy, color='green',
                             linewidth=2, linestyle='--',
                             label=f'Level {level} polygon')
            else:
                self.ax.plot(*poly.exterior.xy, color='black',
                             linewidth=1, linestyle=':', alpha=0.5)

            # ---- NEW: Draw perpendicular bisectors for each level polygon ----
            bisectors = self.get_perpendicular_bisectors(poly)
            bisector_center = self.find_perpendicular_bisector_intersection(poly)
            residual = self.compute_bisector_residual(poly)

            # Draw each perpendicular bisector as a short line segment
            for midpoint, perp_dir, edge_len in bisectors:
                half_len = edge_len * 0.6  # visual length of bisector line
                p_start = midpoint - half_len * perp_dir
                p_end = midpoint + half_len * perp_dir
                self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]],
                             color=color, linewidth=0.8, alpha=0.6, linestyle='-.')

            # Draw the intersection point (circumcenter-like point)
            self.ax.plot(bisector_center[0], bisector_center[1],
                         '*', color=color, markersize=12,
                         label=f'L{level} bisect-int (res={residual:.2f})')

            # Draw a line from the central point to the bisector intersection
            if self.fixed_point_index is not None:
                cp = self.points[self.fixed_point_index]
                self.ax.plot([cp[0], bisector_center[0]], [cp[1], bisector_center[1]],
                             color=color, linewidth=0.5, alpha=0.4)

        # Plot all other points
        for i, point in enumerate(self.points):
            if i == self.fixed_point_index:
                continue
            marker_color = 'gray' if i in self.frozen_points else 'red'
            self.ax.plot(point[0], point[1], 'o', color=marker_color, markersize=5)

        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[2], self.bounds[3])
        self.ax.set_title(f"Radiant Algorithm (Iteration: {self.iteration_count})")
        self.ax.legend(fontsize=7, loc='upper right')

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.click_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def get_neighborhood_polygons(self, idx, max_levels):
        polygons = []
        all_previous = set()

        for level in range(1, max_levels + 1):
            neighbor_indices = self.get_k_level_neighbors(idx, level)
            if not neighbor_indices:
                continue
            new_indices = [i for i in neighbor_indices if i not in all_previous]
            if not new_indices:
                continue
            neighbor_points = [self.points[i] for i in new_indices]
            all_previous.update(new_indices)

            if len(neighbor_points) >= 3:
                center = self.points[idx]
                sorted_points = sorted(neighbor_points,
                                       key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
                polygon = Polygon(sorted_points)
                polygons.append((level, polygon))
            elif len(neighbor_points) == 2:
                line = LineString(neighbor_points)
                polygon = line.buffer(0.1)
                polygons.append((level, polygon))

        return polygons

    def get_k_level_neighbors(self, idx, k_level):
        if k_level < 1:
            return []
        visited = set()
        current_level = {idx}
        for level in range(1, k_level + 1):
            next_level = set()
            for point_idx in current_level:
                if point_idx not in visited:
                    neighbors = self.get_voronoi_neighbors(point_idx)
                    next_level.update(n for n in neighbors if n not in visited)
                    visited.add(point_idx)
            if level < k_level:
                current_level = next_level - visited
        return list(current_level)

    def get_voronoi_neighbors(self, idx):
        neighbors = set()
        for p1, p2 in self.vor.ridge_points:
            if p1 == idx and p2 < len(self.points):
                neighbors.add(p2)
            elif p2 == idx and p1 < len(self.points):
                neighbors.add(p1)
        return list(neighbors)

    # =====================================================================
    #  MODIFIED: progressive_radiant_central_outward — uses bisector intersection
    # =====================================================================

    def progressive_radiant_central_outward(self, tol=1e-6, max_iters=200, lam=0.5):
        if self.fixed_point_index is None:
            messagebox.showinfo("Info", "Run CVT to determine central point first.")
            return

        npts = len(self.points)
        max_levels = self.calculate_max_levels(self.fixed_point_index)
        self.iteration_count = 0

        converged = False
        self.fixed_points = set()

        while not converged:
            converged = True
            central_idx = self.fixed_point_index
            central_pos = self.points[central_idx]

            for level in range(1, max_levels + 1):
                fixed_indices = set()
                for lev_inner in range(1, level):
                    fixed_indices.update(self.get_k_level_neighbors(central_idx, lev_inner))

                level_indices = set(self.get_k_level_neighbors(central_idx, level))
                relax_indices = level_indices - fixed_indices

                if central_idx not in fixed_indices:
                    relax_indices.add(central_idx)

                if not relax_indices:
                    continue

                movement_this_level = float('inf')
                inner_iters = 0

                while movement_this_level > tol and inner_iters < max_iters:
                    inner_iters += 1
                    old_positions = {i: np.array(self.points[i]) for i in relax_indices}

                    centroids = self.calculate_centroids()
                    for idx in relax_indices:
                        if idx == central_idx:
                            continue
                        new_x1 = centroids[idx][0] % self.width
                        new_y1 = centroids[idx][1] % self.height
                        self.points[idx] = (new_x1, new_y1)

                    # MODIFIED: Use bisector intersection instead of polygon centroid
                    if central_idx in relax_indices:
                        level_polygons = self.get_neighborhood_polygons(central_idx, level)
                        poly = None
                        for lvl, p in level_polygons:
                            if lvl == level:
                                poly = p
                                break
                        if poly is not None and not poly.is_empty:
                            # ===== NEW: perpendicular bisector intersection =====
                            radiant_center = self.find_perpendicular_bisector_intersection(poly)
                            lloyd_centroid = centroids[central_idx]
                            new_x = ((1 - lam) * lloyd_centroid[0] + lam * radiant_center[0]) % self.width
                            new_y = ((1 - lam) * lloyd_centroid[1] + lam * radiant_center[1]) % self.height
                            self.points[self.fixed_point_index] = (new_x, new_y)

                    self.vor = Voronoi(self.add_periodic_ghosts(self.points))

                    new_positions = {i: np.array(self.points[i]) for i in relax_indices}
                    movement_this_level = max(
                        np.linalg.norm(new_positions[i] - old_positions[i]) for i in relax_indices)

                    self.iteration_count += 1
                    level_polygons = self.get_neighborhood_polygons(central_idx, level)
                    self.plot_radiant_voronoi(level_polygons)
                    self.root.update_idletasks()

                self.fixed_points.update(relax_indices)

                if movement_this_level > tol:
                    converged = False

            self.fixed_points = set()

        self.info_label.config(text=f"Sweeps Iter: {self.iteration_count}")
        messagebox.showinfo("Info", f"Progressive radiant CVT finished in {self.iteration_count} iterations.")
        final_level_polygons = self.get_neighborhood_polygons(self.fixed_point_index, max_levels)
        self.plot_radiant_voronoi(final_level_polygons)

    def calculate_centroids(self):
        centroids = []
        bounding_box = Polygon([
            (self.bounds[0], self.bounds[2]),
            (self.bounds[1], self.bounds[2]),
            (self.bounds[1], self.bounds[3]),
            (self.bounds[0], self.bounds[3])
        ])
        for i in range(len(self.points)):
            region_index = self.vor.point_region[i]
            region = self.vor.regions[region_index]
            if -1 in region or len(region) == 0:
                centroids.append(self.points[i])
                continue
            polygon = Polygon(self.vor.vertices[region])
            clipped = polygon.intersection(bounding_box)
            if clipped.is_empty:
                centroids.append(self.points[i])
            else:
                centroids.append((clipped.centroid.x, clipped.centroid.y))
        return np.array(centroids)

    def show_original_points(self):
        self.plot_voronoi()


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiGenerator(root)
    root.mainloop()