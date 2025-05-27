import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
import copy
import random
#from sklearn.cluster import KMeans


class VoronoiGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("CVT Generator with Lloyd's Algorithm")

        self.points = []
        self.original_points = []
        self.vor = None
        self.original_vor = None
        self.centroids = None
        self.iteration_count = 0
        self.auto_running = False
        self.auto_previous_points = None  # To track movement between iterations

        self.fixed_point_mode = False
        self.fixed_point_index = None

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.click_canvas = tk.Canvas(self.main_frame, bg='white', width=600, height=400)
        self.click_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        self.select_fixed_button = tk.Button(self.button_frame, text="Select Point A (Off)",
                                             command=self.toggle_fixed_point_mode)
        self.select_fixed_button.pack(fill=tk.X, pady=5)

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
                                        command=self.run_radiant_algorithm)
        self.radiant_button.pack(fill=tk.X, pady=5)

        self.show_original_button = tk.Button(self.button_frame, text="Show Original Points",
                                              command=self.show_original_points)
        self.show_original_button.pack(fill=tk.X, pady=5)

        self.info_label = tk.Label(self.button_frame, text="Iterations: 0")
        self.info_label.pack(fill=tk.X, pady=5)

        self.click_canvas.bind("<Button-1>", self.add_point)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = None

        self.bounds = [0, 600, 0, 400]
        self.width = self.bounds[1] - self.bounds[0]
        self.height = self.bounds[3] - self.bounds[2]

    def toggle_fixed_point_mode(self):
        self.fixed_point_mode = not self.fixed_point_mode
        status = "On" if self.fixed_point_mode else "Off"
        self.select_fixed_button.config(text=f"Select Point A ({status})")

    def add_point(self, event):
        x, y = event.x, self.height - event.y
        self.points.append((x, y))
        self.original_points.append((x, y))

        if self.fixed_point_mode:
            self.fixed_point_index = len(self.points) - 1
            self.click_canvas.create_oval(event.x - 4, event.y - 4, event.x + 4, event.y + 4, fill='blue')
        else:
            self.click_canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill='red')

    def generate_random_points(self):
        seed = simpledialog.askinteger("Random Seed", "Enter random seed (leave empty for random):",
                                       parent=self.root, minvalue=0)
        num_points = simpledialog.askinteger("Number of Points", "How many random points to generate?",
                                             parent=self.root, minvalue=2, initialvalue=20)

        if num_points is None:
            return

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.clear_points()

        # Generate random points
        self.points = [(random.uniform(50, self.width - 50),
                        random.uniform(50, self.height - 50)) for _ in range(num_points)]
        self.original_points = copy.deepcopy(self.points)

        # Draw points on canvas
        self.click_canvas.delete("all")
        for x, y in self.points:
            canvas_y = self.height - y
            self.click_canvas.create_oval(x - 3, canvas_y - 3, x + 3, canvas_y + 3, fill='red')

        # Automatically select the most central point
        self.select_most_central_point()

    def select_most_central_point(self):
        if not self.points:
            return

        # Calculate center of canvas
        center_x, center_y = self.width / 2, self.height / 2

        # Find point closest to center
        min_dist = float('inf')
        central_idx = 0
        for i, (x, y) in enumerate(self.points):
            dist = (x - center_x) ** 2 + (y - center_y) ** 2
            if dist < min_dist:
                min_dist = dist
                central_idx = i

        self.fixed_point_index = central_idx
        self.fixed_point_mode = True
        self.select_fixed_button.config(text="Select Point A (On)")

        # Highlight the central point on canvas
        x, y = self.points[central_idx]
        canvas_y = self.height - y
        self.click_canvas.create_oval(x - 4, canvas_y - 4, x + 4, canvas_y + 4, fill='blue')

    def clear_points(self):
        self.points = []
        self.original_points = []
        self.vor = None
        self.original_vor = None
        self.centroids = None
        self.iteration_count = 0
        self.auto_running = False
        self.fixed_point_index = None
        self.auto_previous_points = None
        self.info_label.config(text="Iterations: 0")
        self.click_canvas.delete("all")
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.ax.clear()

    def add_mirror_points(self, points):
        if len(points) == 0:
            return points

        points = np.array(points)
        mirrored = []

        left_mirror = np.copy(points)
        left_mirror[:, 0] = -left_mirror[:, 0]
        mirrored.append(left_mirror)

        right_mirror = np.copy(points)
        right_mirror[:, 0] = 2 * self.width - right_mirror[:, 0]
        mirrored.append(right_mirror)

        bottom_mirror = np.copy(points)
        bottom_mirror[:, 1] = -bottom_mirror[:, 1]
        mirrored.append(bottom_mirror)

        top_mirror = np.copy(points)
        top_mirror[:, 1] = 2 * self.height - top_mirror[:, 1]
        mirrored.append(top_mirror)

        bl_mirror = np.copy(points)
        bl_mirror[:, 0] = -bl_mirror[:, 0]
        bl_mirror[:, 1] = -bl_mirror[:, 1]
        mirrored.append(bl_mirror)

        tl_mirror = np.copy(points)
        tl_mirror[:, 0] = -tl_mirror[:, 0]
        tl_mirror[:, 1] = 2 * self.height - tl_mirror[:, 1]
        mirrored.append(tl_mirror)

        br_mirror = np.copy(points)
        br_mirror[:, 0] = 2 * self.width - br_mirror[:, 0]
        br_mirror[:, 1] = -br_mirror[:, 1]
        mirrored.append(br_mirror)

        tr_mirror = np.copy(points)
        tr_mirror[:, 0] = 2 * self.width - tr_mirror[:, 0]
        tr_mirror[:, 1] = 2 * self.height - tr_mirror[:, 1]
        mirrored.append(tr_mirror)

        return np.vstack([points] + mirrored)

    def generate_voronoi(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "You need at least 2 points to generate a Voronoi diagram")
            return

        self.points = np.array(self.points)
        all_points = self.add_mirror_points(self.points)
        self.vor = Voronoi(all_points)
        self.original_vor = copy.deepcopy(self.vor)
        self.centroids = None
        self.iteration_count = 0
        self.auto_previous_points = None
        self.info_label.config(text="Iterations: 0")

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

        # === Color point A and its neighbors ===
        color_map = ['orange', 'purple', 'green']
        plotted_indices = set()

        if self.fixed_point_index is not None:
            self.ax.plot(self.points[self.fixed_point_index][0],
                         self.points[self.fixed_point_index][1],
                         'o', color='blue', markersize=8, label='Point A')
            plotted_indices.add(self.fixed_point_index)

            # Compute neighbors
            neighborhood_levels = self.get_neighborhood_polygons(self.fixed_point_index, max_levels=3)

            for level, polygon in neighborhood_levels:
                color = color_map[level % len(color_map)]
                level_neighbors = set()

                # Extract indices from the polygon's coordinates
                for idx, pt in enumerate(self.points):
                    p = Point(pt)
                    if polygon.contains(p) or polygon.distance(p) < 1e-2:
                        if idx != self.fixed_point_index:
                            level_neighbors.add(idx)

                for idx in level_neighbors:
                    if idx not in plotted_indices:
                        self.ax.plot(self.points[idx][0], self.points[idx][1], 'o',
                                     color=color, markersize=6)
                        plotted_indices.add(idx)

        # === Plot the rest ===
        for i, point in enumerate(self.points):
            if i not in plotted_indices:
                self.ax.plot(point[0], point[1], 'o', color='red', markersize=5)

        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[2], self.bounds[3])
        self.ax.set_title(f"Voronoi Diagram (Iteration: {self.iteration_count})")

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.click_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
            if level < k_level:  # Only keep last level's new neighbors
                current_level = next_level - visited

        return list(current_level)

    def get_neighborhood_polygons(self, idx, max_levels):
        polygons = []
        all_previous = set()

        for level in range(1, max_levels + 1):
            neighbor_indices = self.get_k_level_neighbors(idx, level)

            if not neighbor_indices:
                continue

            # Skip if we've already processed these points in previous levels
            new_indices = [i for i in neighbor_indices if i not in all_previous]
            if not new_indices:
                continue

            neighbor_points = [self.points[i] for i in new_indices]
            all_previous.update(new_indices)

            # Create a polygon by connecting all points in order (may need sorting)
            if len(neighbor_points) >= 3:
                # Option 1: Sort points by angle relative to center point
                center = self.points[idx]
                sorted_points = sorted(neighbor_points,
                                       key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
                polygon = Polygon(sorted_points)
                polygons.append((level, polygon))
            elif len(neighbor_points) == 2:
                # For just 2 points, create a line (which we'll treat as a degenerate polygon)
                line = LineString(neighbor_points)
                # Buffer it slightly to make it a valid polygon
                polygon = line.buffer(0.1)
                polygons.append((level, polygon))

        return polygons

    def auto_lloyd(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Generate the Voronoi diagram first.")
            return

        if self.auto_running:
            self.auto_running = False
            self.auto_lloyd_button.config(text="Auto CVT")
            return
        else:
            self.auto_running = True
            self.auto_lloyd_button.config(text="Stop Auto CVT")
            self.auto_previous_points = self.points.copy()
            self.auto_iterations = 0
            self.auto_max_iterations = 100
            self.auto_tolerance = 1e-3
            self.root.after(10, self.auto_lloyd_iteration)

    def auto_lloyd_iteration(self):
        if not self.auto_running or self.auto_iterations >= self.auto_max_iterations:
            self.auto_running = False
            self.auto_lloyd_button.config(text="Auto CVT")
            return

        centroids = self.calculate_centroids()
        movement = np.linalg.norm(self.auto_previous_points - centroids)

        if movement < self.auto_tolerance:
            self.auto_running = False
            self.auto_lloyd_button.config(text="Auto CVT")
            return

        all_points = self.add_mirror_points(centroids)
        self.vor = Voronoi(all_points)
        self.points = centroids.copy()
        self.auto_previous_points = centroids.copy()
        self.iteration_count += 1
        self.auto_iterations += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

        self.root.after(10, self.auto_lloyd_iteration)

    def get_voronoi_neighbors(self, idx):
        neighbors = set()
        for p1, p2 in self.vor.ridge_points:
            if p1 == idx and p2 < len(self.points):
                neighbors.add(p2)
            elif p2 == idx and p1 < len(self.points):
                neighbors.add(p1)
        return list(neighbors)

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

    def run_radiant_algorithm(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Generate the Voronoi diagram first.")
            return

        if self.fixed_point_index is None:
            messagebox.showwarning("Warning", "Select a central point first.")
            return

        # First run Lloyd's algorithm until convergence
        self.auto_lloyd()
        self.root.after(100, self._continue_radiant_after_lloyd)

    def _continue_radiant_after_lloyd(self):
        if self.auto_running:  # If Lloyd's is still running
            self.root.after(100, self._continue_radiant_after_lloyd)
            return

        # Now apply the radiant algorithm
        self.apply_radiant_algorithm()

    def apply_lloyd(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Generate the Voronoi diagram first.")
            return

        self.centroids = self.calculate_centroids()
        all_points = self.add_mirror_points(self.centroids)
        self.vor = Voronoi(all_points)
        self.points = self.centroids.copy()
        self.iteration_count += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

    def apply_radiant_algorithm(self):
        if self.fixed_point_index is None:
            messagebox.showwarning("Warning", "Select a central point first.")
            return

        if self.vor is None:
            messagebox.showwarning("Warning", "Generate the Voronoi diagram first.")
            return

        # Get neighborhood polygons for the central point
        level_polygons = self.get_neighborhood_polygons(self.fixed_point_index, 5)

        if not level_polygons:
            return

        # Calculate centroids for each level's polygon
        level_centroids = [poly.centroid for level, poly in level_polygons]

        # Calculate centroid of all level centroids
        from shapely.geometry import MultiPoint
        centroids_all = MultiPoint(level_centroids).centroid

        # Get the current position of the central point
        current_pos = self.points[self.fixed_point_index]

        # Move the central point towards the overall centroid
        lambda_ = 0.7  # Higher weight for radiant effect
        new_x = (1 - lambda_) * current_pos[0] + lambda_ * centroids_all.x
        new_y = (1 - lambda_) * current_pos[1] + lambda_ * centroids_all.y

        # Update the point position
        self.points[self.fixed_point_index] = (new_x, new_y)

        # Regenerate Voronoi diagram
        self.vor = Voronoi(self.add_mirror_points(self.points))
        self.iteration_count += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")

        # Plot the result with neighborhood visualization
        self.plot_voronoi_with_neighborhoods(level_polygons)

    def plot_voronoi_with_neighborhoods(self, level_polygons):
        self.ax.clear()

        bounding_box = Polygon([
            (self.bounds[0], self.bounds[2]),
            (self.bounds[1], self.bounds[2]),
            (self.bounds[1], self.bounds[3]),
            (self.bounds[0], self.bounds[3])
        ])

        # Plot Voronoi cells
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

            # Visualize neighborhood polygons in black
            for level, poly in level_polygons:
                self.ax.plot(*poly.exterior.xy, color='black',
                             linewidth=2, linestyle='--',
                             label=f'Level {level} neighborhood')

        # Plot all points
        for i, point in enumerate(self.points):
            if i == self.fixed_point_index:
                continue
            self.ax.plot(point[0], point[1], 'o', color='red', markersize=5)

        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[2], self.bounds[3])
        self.ax.set_title(f"Voronoi Diagram (Iteration: {self.iteration_count})")
        self.ax.legend()

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.click_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiGenerator(root)
    root.mainloop()