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

        for i, point in enumerate(self.points):
            color = 'blue' if i == self.fixed_point_index else 'red'
            self.ax.plot(point[0], point[1], 'o', color=color)

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

        centroids = self.calculate_centroids()

        if self.fixed_point_index is not None:
            neighbors = self.get_voronoi_neighbors(self.fixed_point_index)
            if neighbors:
                neighbor_points = [self.points[i] for i in neighbors if i < len(self.points)]

                if neighbor_points:
                    from shapely.geometry import MultiPoint
                    neighbor_multipoint = MultiPoint(neighbor_points)
                    poly_centroid = neighbor_multipoint.centroid

                    lambda_ = 0.5
                    current_centroid = centroids[self.fixed_point_index]
                    new_x = (1 - lambda_) * current_centroid[0] + lambda_ * poly_centroid.x
                    new_y = (1 - lambda_) * current_centroid[1] + lambda_ * poly_centroid.y
                    centroids[self.fixed_point_index] = (new_x, new_y)

        self.auto_previous_points = np.array(self.points)
        self.points = centroids
        self.vor = Voronoi(self.add_mirror_points(self.points))
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
                return

        # Schedule next iteration
        self.root.after(100, self.run_auto_lloyd)

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