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

        # Initialize variables
        self.points = []
        self.original_points = []
        self.vor = None
        self.centroids = None
        self.iteration_count = 0
        self.auto_running = False
        self.auto_previous_points = None
        self.radiant_running = False
        self.radiant_iterations = 0
        self.max_radiant_iterations = 100  # Set maximum radiant iterations
        self.fixed_point_index = None

        # Setup UI
        self.setup_ui()

        # Canvas bounds
        self.bounds = [0, 600, 0, 400]
        self.width = self.bounds[1] - self.bounds[0]
        self.height = self.bounds[3] - self.bounds[2]

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for points
        self.click_canvas = tk.Canvas(self.main_frame, bg='white', width=600, height=400)
        self.click_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Button frame
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Buttons
        buttons = [
            ("Generate Random Points", self.generate_random_points),
            ("Clear Points", self.clear_points),
            ("Generate Voronoi", self.generate_voronoi),
            ("Lloyd's Algorithm", self.apply_lloyd),
            ("Auto CVT", self.auto_lloyd),
            ("Run Radiant Algorithm", self.run_radiant_algorithm),
            ("Show Original Points", self.show_original_points)
        ]

        for text, command in buttons:
            btn = tk.Button(self.button_frame, text=text, command=command)
            btn.pack(fill=tk.X, pady=5)

        # Info label
        self.info_label = tk.Label(self.button_frame, text="Iterations: 0")
        self.info_label.pack(fill=tk.X, pady=5)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = None

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
            messagebox.showwarning("Warning", "Need at least 2 points")
            return

        self.points = np.array(self.points)
        self.vor = Voronoi(self.add_mirror_points(self.points))
        self.iteration_count = 0
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

    def apply_lloyd(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Generate Voronoi first")
            return

        self.centroids = self.calculate_centroids()
        self.vor = Voronoi(self.add_mirror_points(self.centroids))
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
        self.apply_lloyd()

        if self.auto_previous_points is not None:
            movement = np.linalg.norm(self.auto_previous_points - self.points)
            if movement < self.auto_tolerance:
                self.auto_running = False
                self.auto_lloyd_button.config(text="Auto CVT")
                self.select_most_central_point()
                return

        self.auto_previous_points = current_points
        self.root.after(100, self.run_auto_lloyd)

    def select_most_central_point(self):
        if not self.points:
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

        # Highlight central point
        self.plot_voronoi()
        self.ax.plot(self.points[central_idx][0], self.points[central_idx][1],
                     'o', color='blue', markersize=8, label='Central Point')
        self.ax.legend()
        self.canvas.draw()

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

    def apply_radiant_algorithm(self):
        level_polygons = self.get_neighborhood_polygons(self.fixed_point_index, 3)

        if not level_polygons:
            return

        level_centroids = [poly.centroid for level, poly in level_polygons]
        centroids_all = MultiPoint(level_centroids).centroid

        current_pos = self.points[self.fixed_point_index]
        lambda_ = 0.7  # Radiant effect weight
        new_x = (1 - lambda_) * current_pos[0] + lambda_ * centroids_all.x
        new_y = (1 - lambda_) * current_pos[1] + lambda_ * centroids_all.y

        self.points[self.fixed_point_index] = (new_x, new_y)
        self.vor = Voronoi(self.add_mirror_points(self.points))
        self.iteration_count += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")

        self.plot_radiant_voronoi(level_polygons)

    def plot_radiant_voronoi(self, level_polygons):
        self.ax.clear()

        # Plot Voronoi cells
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

            # Plot neighborhood polygons in black
            for level, poly in level_polygons:
                self.ax.plot(*poly.exterior.xy, color='black',
                             linewidth=2, linestyle='--',
                             label=f'Level {level} neighborhood')

        # Plot all other points
        for i, point in enumerate(self.points):
            if i == self.fixed_point_index:
                continue
            self.ax.plot(point[0], point[1], 'o', color='red', markersize=5)

        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[2], self.bounds[3])
        self.ax.set_title(f"Radiant Algorithm (Iteration: {self.iteration_count})")
        self.ax.legend()

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.click_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ... [Keep all the helper methods like add_mirror_points, calculate_centroids,
    # get_neighborhood_polygons, etc. from previous version] ...


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiGenerator(root)
    root.mainloop()