import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import Voronoi, distance
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.ops import unary_union
import alphashape
from descartes import PolygonPatch
import copy


class RadiantVoronoiGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("CVT Generator with Radiant Algorithm")

        # Initialize variables
        self.points = []
        self.original_points = []
        self.vor = None
        self.original_vor = None
        self.centroids = None
        self.iteration_count = 0
        self.auto_running = False
        self.auto_previous_points = None
        self.cvt_converged = False
        self.seed = 42
        self.fixed_point_mode = False
        self.fixed_point_index = None
        self.neighborhood_polygons = []

        # Setup GUI
        self.setup_gui()
        self.setup_matplotlib()

    def setup_gui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.click_canvas = tk.Canvas(self.main_frame, bg='white', width=600, height=400)
        self.click_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Control buttons
        controls = [
            ("Select Point A (Off)", self.toggle_fixed_point_mode),
            ("Random Generate", lambda: self.random_generate(20)),
            ("Clear Points", self.clear_points),
            ("Generate Voronoi", self.generate_voronoi),
            ("Lloyd's Algorithm", self.apply_lloyd),
            ("Auto CVT", self.auto_lloyd),
            ("Run Radiant", lambda: self.run_radiant(max_iter=150)),
            ("Show Original Points", self.show_original_points)
        ]

        for text, command in controls:
            button = tk.Button(self.button_frame, text=text, command=command)
            button.pack(fill=tk.X, pady=5)

        self.info_label = tk.Label(self.button_frame, text="Iterations: 0")
        self.info_label.pack(fill=tk.X, pady=5)
        self.click_canvas.bind("<Button-1>", self.add_point)

    def setup_matplotlib(self):
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

    def random_generate(self, num_points=20):
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.clear_points()
        for _ in range(num_points):
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            self.points.append((x, y))
            self.original_points.append((x, y))

            canvas_x = x
            canvas_y = self.height - y
            self.click_canvas.create_oval(canvas_x - 3, canvas_y - 3, canvas_x + 3, canvas_y + 3, fill='red')

        self.generate_voronoi()
        self.auto_select_center_point()

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
        self.cvt_converged = False
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
        self.cvt_converged = False
        self.info_label.config(text="Iterations: 0")

        self.plot_voronoi()

    def get_neighborhood_polygon(self, point_indices):
        """Create alpha shape polygon for neighborhood points"""
        if len(point_indices) < 3:
            return None

        points = np.array([self.points[i] for i in point_indices])
        try:
            alpha = max(0.1, min(2.0, 1.0 + len(points) / 20))  # Dynamic alpha
            polygon = alphashape.alphashape(points, alpha)
            return polygon if polygon.is_valid else None
        except:
            return None

    def get_neighborhood_polygons(self, idx, max_levels):
        """Get concave polygons for each neighborhood level"""
        polygons = []
        all_previous = set()

        for level in range(1, max_levels + 1):
            neighbor_indices = self.get_k_level_neighbors(idx, level)
            neighbor_indices = [i for i in neighbor_indices if i not in all_previous]

            if len(neighbor_indices) >= 3:
                polygon = self.get_neighborhood_polygon(neighbor_indices)
                if polygon:
                    polygons.append((level, polygon))
                    all_previous.update(neighbor_indices)

        return polygons

    def plot_voronoi(self):
        """Enhanced plotting with alpha shape visualization"""
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
                self.ax.fill(*clipped.exterior.xy, alpha=0.3)
            elif isinstance(clipped, MultiPolygon):
                for poly in clipped.geoms:
                    self.ax.fill(*poly.exterior.xy, alpha=0.3)

        # Plot points
        for i, point in enumerate(self.points):
            color = 'blue' if i == self.fixed_point_index else 'red'
            self.ax.plot(point[0], point[1], 'o', color=color, markersize=6)

        # Plot neighborhood polygons if they exist
        if hasattr(self, 'neighborhood_polygons') and self.neighborhood_polygons:
            colors = ['#4CAF50', '#FF9800', '#9C27B0']  # Green, Orange, Purple
            for level, poly in self.neighborhood_polygons:
                patch = PolygonPatch(poly, fc=colors[level - 1], ec=colors[level - 1],
                                     alpha=0.3, linewidth=1.5)
                self.ax.add_patch(patch)
                self.ax.plot(poly.centroid.x, poly.centroid.y, 'o',
                             color=colors[level - 1], markersize=8)

        self.ax.set_xlim(self.bounds[0], self.bounds[1])
        self.ax.set_ylim(self.bounds[2], self.bounds[3])
        self.ax.set_title(f"Voronoi Diagram (Iteration: {self.iteration_count})")

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.click_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def apply_radiant_optimization(self, levels=3, lambda_=0.7):
        """Radiant optimization with concave polygons"""
        centroids = self.calculate_centroids()
        self.neighborhood_polygons = self.get_neighborhood_polygons(self.fixed_point_index, levels)

        if self.neighborhood_polygons:
            level_centroids = [poly.centroid for level, poly in self.neighborhood_polygons]
            centroids_all = MultiPoint(level_centroids).centroid

            lloyd_pos = centroids[self.fixed_point_index]
            new_x = (1 - lambda_) * lloyd_pos[0] + lambda_ * centroids_all.x
            new_y = (1 - lambda_) * lloyd_pos[1] + lambda_ * centroids_all.y
            centroids[self.fixed_point_index] = (new_x, new_y)

            self.points = centroids
            self.vor = Voronoi(self.add_mirror_points(self.points))
            self.plot_voronoi()

    # [Keep all other existing methods unchanged]


if __name__ == "__main__":
    root = tk.Tk()
    app = RadiantVoronoiGenerator(root)
    root.mainloop()