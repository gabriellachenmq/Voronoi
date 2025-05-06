import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.spatial import Voronoi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import matplotlib.patches as mpatches
import shapely.ops as ops


class VoronoiGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("CVT Generator with Lloyd's Algorithm")

        self.points = []
        self.original_points = []
        self.vor = None
        self.centroids = None
        self.iteration_count = 0

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

        self.show_original_button = tk.Button(self.button_frame, text="Show Original Points", command=self.show_original_points)
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
        self.centroids = None
        self.iteration_count = 0
        self.info_label.config(text="Iterations: 0")
        self.click_canvas.delete("all")
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.ax.clear()

    def add_mirror_points(self, points):
        """Add mirrored points around the canvas bounds to ensure bounded Voronoi regions"""
        width = self.bounds[1]
        height = self.bounds[3]

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

        self.points = np.array(self.points)
        self.original_points_array = self.points.copy()  # Save for display and centroid use
        all_points = self.add_mirror_points(self.points)
        self.vor = Voronoi(all_points)
        self.centroids = None
        self.iteration_count = 0
        self.info_label.config(text="Iterations: 0")

        self.plot_voronoi()

    def bounded_voronoi_polygons(self, vor, bounds):
        from shapely.geometry import LineString

        center = vor.points.mean(axis=0)
        radius = np.max(np.linalg.norm(vor.points - center, axis=1)) * 2

        # Corners of the bounding box
        min_x, max_x, min_y, max_y = bounds
        bbox = Polygon([
            (min_x, min_y), (max_x, min_y),
            (max_x, max_y), (min_x, max_y)
        ])

        regions = []
        for region_index in vor.point_region:
            region = vor.regions[region_index]
            if not region or -1 in region:
                # Reconstruct infinite region
                point_index = vor.point_region.tolist().index(region_index)
                region_points = []

                for p1, p2 in zip(vor.ridge_points, vor.ridge_vertices):
                    if point_index in p1 and -1 in p2:
                        i = p1[0] if p1[1] == point_index else p1[1]
                        t = vor.points[point_index] - vor.points[i]
                        t = t / np.linalg.norm(t)
                        n = np.array([-t[1], t[0]])  # normal
                        midpoint = (vor.points[point_index] + vor.points[i]) / 2
                        far_point = midpoint + n * radius
                        region_points.append(far_point)

                finite_vertices = [vor.vertices[v] for v in region if v != -1]
                full_region = finite_vertices + region_points
                polygon = Polygon(full_region)
            else:
                polygon = Polygon([vor.vertices[i] for i in region])

            clipped = polygon.intersection(bbox)
            if clipped.is_empty:
                regions.append(None)
            else:
                regions.append(clipped)

        return regions

    def plot_voronoi(self):
        self.ax.clear()
        bounding_box = Polygon([
            (self.bounds[0], self.bounds[2]),
            (self.bounds[1], self.bounds[2]),
            (self.bounds[1], self.bounds[3]),
            (self.bounds[0], self.bounds[3])
        ])

        # Plot clipped Voronoi cells for original points only
        num_real = len(self.original_points_array)
        regions = self.bounded_voronoi_polygons(self.vor, self.bounds)
        for i, poly in enumerate(regions[:num_real]):
            if poly and poly.geom_type == "Polygon":
                patch = mpatches.Polygon(np.array(poly.exterior.coords), edgecolor='blue', facecolor='none')
                self.ax.add_patch(patch)

        # Draw arrows from original to centroid if available
        if self.centroids is not None:
            for i in range(num_real):
                dx = self.centroids[i, 0] - self.vor.points[i, 0]
                dy = self.centroids[i, 1] - self.vor.points[i, 1]
                self.ax.arrow(self.vor.points[i, 0], self.vor.points[i, 1], dx, dy,
                              head_width=5, head_length=5, fc='green', ec='green')
            self.ax.plot(self.vor.points[:num_real, 0], self.vor.points[:num_real, 1], 'ro')
            self.ax.plot(self.centroids[:, 0], self.centroids[:, 1], 'go')
        else:
            self.ax.plot(self.vor.points[:num_real, 0], self.vor.points[:num_real, 1], 'ro')

        self.ax.set_title(f'Voronoi Diagram (Iteration: {self.iteration_count})')
        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[2], self.bounds[3])
        self.ax.set_aspect('equal')

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
        self.vor = Voronoi(self.centroids)
        self.points = self.centroids.copy()
        self.iteration_count += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

    def auto_lloyd(self):
        tolerance = 1e-3
        max_iter = 100
        for _ in range(max_iter):
            centroids = self.calculate_centroids()
            movement = np.linalg.norm(self.points - centroids)
            self.points = centroids
            self.vor = Voronoi(self.points)
            self.iteration_count += 1
            if movement < tolerance:
                break
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

    def calculate_centroids(self):
        """Only compute centroids for the original points"""
        num_real = len(self.original_points_array)
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
                centroids[i] = self.vor.points[i]
                continue

            polygon = Polygon(self.vor.vertices[region])
            clipped = polygon.intersection(bounding_box)
            if clipped.is_empty:
                centroids[i] = self.vor.points[i]
            else:
                c = clipped.centroid
                centroids[i] = [c.x, c.y]
        return centroids

    def show_original_points(self):
        self.plot_voronoi(show_original=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiGenerator(root)
    root.mainloop()
