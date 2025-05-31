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

        # Initialize all required attributes
        self.points = []
        self.original_points = []
        self.vor = None
        self.centroids = None
        self.iteration_count = 0
        self.auto_running = False
        self.radiant_running = False
        self.radiant_iterations = 0
        self.max_radiant_iterations = 100
        self.fixed_point_index = None
        self.auto_previous_points = None  # Initialize this attribute
        self.auto_tolerance = 1e-3  # Initialize tolerance

        # Create UI elements
        self.create_ui()

        # Canvas dimensions
        self.bounds = [0, 600, 0, 400]
        self.width = self.bounds[1] - self.bounds[0]
        self.height = self.bounds[3] - self.bounds[2]

    def create_ui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for points
        self.click_canvas = tk.Canvas(self.main_frame, bg='white', width=600, height=400)
        self.click_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Button frame
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Create and store references to all buttons
        self.random_button = tk.Button(self.button_frame, text="Generate Random Points",
                                     command=self.generate_random_points)
        self.random_button.pack(fill=tk.X, pady=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear Points",
                                    command=self.clear_points)
        self.clear_button.pack(fill=tk.X, pady=5)

        self.generate_button = tk.Button(self.button_frame, text="Generate Voronoi",
                                       command=self.generate_voronoi)
        self.generate_button.pack(fill=tk.X, pady=5)

        self.lloyd_button = tk.Button(self.button_frame, text="Lloyd's Algorithm",
                                     command=self.apply_lloyd)
        self.lloyd_button.pack(fill=tk.X, pady=5)

        self.auto_lloyd_button = tk.Button(self.button_frame, text="Auto CVT",
                                          command=self.auto_lloyd)
        self.auto_lloyd_button.pack(fill=tk.X, pady=5)

        self.radiant_button = tk.Button(self.button_frame, text="Run Radiant Algorithm",
                                       command=self.run_radiant_algorithm)
        self.radiant_button.pack(fill=tk.X, pady=5)

        self.show_original_button = tk.Button(self.button_frame, text="Show Original Points",
                                            command=self.show_original_points)
        self.show_original_button.pack(fill=tk.X, pady=5)

        # Info label
        self.info_label = tk.Label(self.button_frame, text="Iterations: 0")
        self.info_label.pack(fill=tk.X, pady=5)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = None

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

        # Generate well-distributed points using KMeans
        initial_points = np.random.rand(num_points * 10, 2)
        initial_points[:, 0] = initial_points[:, 0] * (self.width - 100) + 50
        initial_points[:, 1] = initial_points[:, 1] * (self.height - 100) + 50

        kmeans = KMeans(n_clusters=num_points, random_state=seed if seed else None)
        kmeans.fit(initial_points)
        self.points = kmeans.cluster_centers_.tolist()
        self.original_points = copy.deepcopy(self.points)

        # Draw points
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

    def generate_voronoi(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "You need at least 2 points to generate a Voronoi diagram")
            return

        self.points = np.array(self.points)
        all_points = self.add_mirror_points(self.points)
        self.vor = Voronoi(all_points)
        self.iteration_count = 0
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

    def add_mirror_points(self, points):
        if len(points) == 0:
            return points

        points = np.array(points)
        mirrored = []

        # Basic mirroring (left, right, top, bottom)
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

        return np.vstack([points] + mirrored)

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

        # Plot all points
        for i, point in enumerate(self.points):
            color = 'blue' if i == self.fixed_point_index else 'red'
            self.ax.plot(point[0], point[1], 'o', color=color, markersize=5)

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
            messagebox.showwarning("Warning", "Generate the Voronoi diagram first.")
            return

        self.centroids = self.calculate_centroids()
        all_points = self.add_mirror_points(self.centroids)
        self.vor = Voronoi(all_points)
        self.points = self.centroids.copy()
        self.iteration_count += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

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
            self.root.after(10, self.run_auto_lloyd)

    def run_auto_lloyd(self):
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

        self.root.after(10, self.run_auto_lloyd)

    def select_most_central_point(self):
        if len(self.points):
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

    def run_radiant_algorithm(self):
        if self.fixed_point_index is None:
            messagebox.showwarning("Warning", "No central point selected. Run CVT first.")
            return

        if self.vor is None:
            messagebox.showwarning("Warning", "Generate the Voronoi diagram first.")
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

        # Get neighborhood polygons
        level_polygons = self.get_neighborhood_polygons(self.fixed_point_index, 3)

        if level_polygons:
            # Calculate centroids for each level's polygon
            level_centroids = [poly.centroid for level, poly in level_polygons]
            centroids_all = MultiPoint(level_centroids).centroid

            # Move central point
            current_pos = self.points[self.fixed_point_index]
            lambda_ = 0.7
            new_x = (1 - lambda_) * current_pos[0] + lambda_ * centroids_all.x
            new_y = (1 - lambda_) * current_pos[1] + lambda_ * centroids_all.y
            self.points[self.fixed_point_index] = (new_x, new_y)

            # Regenerate Voronoi
            self.vor = Voronoi(self.add_mirror_points(self.points))
            self.iteration_count += 1
            self.radiant_iterations += 1
            self.info_label.config(
                text=f"Iterations: {self.iteration_count} (Radiant: {self.radiant_iterations})")

            # Visualize
            self.plot_radiant_voronoi(level_polygons)

        # Schedule next iteration
        self.root.after(100, self.run_radiant_iteration)

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