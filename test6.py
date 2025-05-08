import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon
import copy

class VoronoiGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("CVT Generator with Lloyd's Algorithm")

        self.points = []
        self.original_points = []
        self.vor = None
        self.original_vor = None  # Store original Voronoi diagram
        self.centroids = None
        self.iteration_count = 0
        self.auto_running = False

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.click_canvas = tk.Canvas(self.main_frame, bg='white', width=600, height=400)
        self.click_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear Points", command=self.clear_points)
        self.clear_button.pack(fill=tk.X, pady=5)

        self.generate_button = tk.Button(self.button_frame, text="Generate Voronoi", command=self.generate_voronoi)
        self.generate_button.pack(fill=tk.X, pady=5)

        self.lloyd_button = tk.Button(self.button_frame, text="Lloyd's Algorithm", command=self.apply_lloyd)
        self.lloyd_button.pack(fill=tk.X, pady=5)

        self.auto_lloyd_button = tk.Button(self.button_frame, text="Auto CVT", command=self.auto_lloyd)
        self.auto_lloyd_button.pack(fill=tk.X, pady=5)

        self.show_original_button = tk.Button(self.button_frame, text="Show Original Points",
                                              command=self.show_original_points)
        self.show_original_button.pack(fill=tk.X, pady=5)

        self.info_label = tk.Label(self.button_frame, text="Iterations: 0")
        self.info_label.pack(fill=tk.X, pady=5)

        self.click_canvas.bind("<Button-1>", self.add_point)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = None

        self.bounds = [0, 600, 0, 400]

    def add_point(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        self.original_points.append((x, y))
        self.click_canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill='red')

    def clear_points(self):
        self.points = []
        self.original_points = []
        self.vor = None
        self.original_vor = None
        self.centroids = None
        self.iteration_count = 0
        self.auto_running = False
        self.info_label.config(text="Iterations: 0")
        self.click_canvas.delete("all")
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.ax.clear()

    def add_mirror_points(self, points):
        """Add mirrored points around the canvas bounds to ensure bounded Voronoi regions"""
        if len(points) == 0:
            return points

        points = np.array(points)
        width = self.bounds[1] - self.bounds[0]
        height = self.bounds[3] - self.bounds[2]

        mirror_offsets = [
            (-width, 0), (width, 0),
            (0, -height), (0, height),
            (-width, -height), (-width, height),
            (width, -height), (width, height),
        ]

        mirrored = []
        for dx, dy in mirror_offsets:
            mirrored.extend(points + np.array([dx, dy]))

        return np.vstack([points] + mirrored)

    def generate_voronoi(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "You need at least 2 points to generate a Voronoi diagram")
            return

        # If we already have an original Voronoi diagram, restore it
        if self.original_vor is not None:
            self.vor = copy.deepcopy(self.original_vor)
            self.points = np.array(self.original_points)
            self.iteration_count = 0
            self.info_label.config(text="Iterations: 0")
            self.plot_voronoi()
            return

        # Otherwise generate new Voronoi diagram
        self.points = np.array(self.points)
        all_points = self.add_mirror_points(self.points)
        self.vor = Voronoi(all_points)
        self.original_vor = copy.deepcopy(self.vor)  # Store original state
        self.centroids = None
        self.iteration_count = 0
        self.info_label.config(text="Iterations: 0")

        self.plot_voronoi()

    def plot_voronoi(self, show_original=False):
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
            if clipped.geom_type == 'Polygon':
                self.ax.fill(*clipped.exterior.xy, alpha=0.4)
            elif clipped.geom_type == 'MultiPolygon':
                for p in clipped.geoms:
                    self.ax.fill(*p.exterior.xy, alpha=0.4)

        self.ax.plot(self.points[:, 0], self.points[:, 1], 'ro', label='Current Points')
        if show_original:
            original = np.array(self.original_points)
            self.ax.plot(original[:, 0], original[:, 1], 'bo', label='Original Points')

        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[2], self.bounds[3])
        self.ax.set_title(f"Voronoi Diagram (Iteration: {self.iteration_count})")
        self.ax.legend()

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.click_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.click_canvas.delete("all")

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

    def calculate_centroids(self):
        num_real = len(self.points)
        centroids = np.zeros((num_real, 2))

        bounding_box = Polygon([
            (self.bounds[0], self.bounds[2]),
            (self.bounds[1], self.bounds[2]),
            (self.bounds[1], self.bounds[3]),
            (self.bounds[0], self.bounds[3])
        ])

        for i in range(num_real):
            region_index = self.vor.point_region[i]
            region = self.vor.regions[region_index]

            if -1 in region or len(region) == 0:
                centroids[i] = self.points[i]
                continue

            polygon = Polygon(self.vor.vertices[region])
            clipped = polygon.intersection(bounding_box)
            if clipped.is_empty:
                centroids[i] = self.points[i]
            else:
                c = clipped.centroid
                centroids[i] = [c.x, c.y]
        return centroids

    def show_original_points(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Generate the Voronoi diagram first.")
            return
        self.plot_voronoi(show_original=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiGenerator(root)
    root.mainloop()