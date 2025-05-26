import tkinter as tk
from tkinter import messagebox
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import Voronoi, distance
from shapely.geometry import Polygon, MultiPolygon, MultiPoint, Point, LineString
from shapely.ops import polygonize, unary_union
import copy
import alphashape
from descartes import PolygonPatch


class RadiantVoronoiGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("CVT Generator with Radiant Algorithm")

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

        # Main GUI setup
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.click_canvas = tk.Canvas(self.main_frame, bg='white', width=600, height=400)
        self.click_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Control buttons
        self.select_fixed_button = tk.Button(self.button_frame, text="Select Point A (Off)",
                                             command=self.toggle_fixed_point_mode)
        self.select_fixed_button.pack(fill=tk.X, pady=5)

        self.random_button = tk.Button(self.button_frame, text="Random Generate",
                                       command=lambda: self.random_generate(20))
        self.random_button.pack(fill=tk.X, pady=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear Points", command=self.clear_points)
        self.clear_button.pack(fill=tk.X, pady=5)

        self.generate_button = tk.Button(self.button_frame, text="Generate Voronoi", command=self.generate_voronoi)
        self.generate_button.pack(fill=tk.X, pady=5)

        self.lloyd_button = tk.Button(self.button_frame, text="Lloyd's Algorithm", command=self.apply_lloyd)
        self.lloyd_button.pack(fill=tk.X, pady=5)

        self.auto_lloyd_button = tk.Button(self.button_frame, text="Auto CVT", command=self.auto_lloyd)
        self.auto_lloyd_button.pack(fill=tk.X, pady=5)

        self.radiant_button = tk.Button(self.button_frame, text="Run Radiant", command=self.run_radiant)
        self.radiant_button.pack(fill=tk.X, pady=5)

        self.show_original_button = tk.Button(self.button_frame, text="Show Original Points",
                                              command=self.show_original_points)
        self.show_original_button.pack(fill=tk.X, pady=5)

        self.info_label = tk.Label(self.button_frame, text="Iterations: 0")
        self.info_label.pack(fill=tk.X, pady=5)

        self.click_canvas.bind("<Button-1>", self.add_point)

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = None

        # Bounds and dimensions
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
                self.cvt_converged = True
                return

        self.root.after(100, self.run_auto_lloyd)

    def run_radiant(self, levels=3, lambda_=0.5, max_iter=300):
        if len(self.points) == 0:
            self.random_generate()

        self.run_to_cvt()

        self.auto_select_center_point()

        if self.fixed_point_index is not None:
            for iteration in range(max_iter):
                self.apply_radiant_optimization(levels, lambda_)

                # if iteration > 10 and self.check_convergence():
                    # break

                self.root.update()
                self.root.after(50)

            messagebox.showinfo("Radiant Complete",
                                f"Finished {min(iteration + 1, max_iter)} Radiant iterations")
        else:
            messagebox.showwarning("Warning", "No center point selected")

    def check_convergence(self, tolerance=1e-4):
        if not hasattr(self, 'previous_radiant_position'):
            self.previous_radiant_position = self.points[self.fixed_point_index].copy()
            return False

        movement = np.linalg.norm(self.points[self.fixed_point_index] -
                                  self.previous_radiant_position)
        self.previous_radiant_position = self.points[self.fixed_point_index].copy()
        return movement < tolerance

    def run_to_cvt(self, tolerance=1e-3, max_iter=200):
        """Run Lloyd's until convergence, without center point selection"""
        self.cvt_converged = False
        prev_points = None

        for _ in range(max_iter):
            self.apply_lloyd()
            current_points = np.array(self.points)

            if prev_points is not None:
                movement = np.linalg.norm(current_points - prev_points)
                if movement < tolerance:
                    self.cvt_converged = True
                    self.info_label.config(text=f"CVT Converged ({self.iteration_count} iterations)")
                    break

            prev_points = current_points.copy()
            self.root.update()
            self.root.after(100)  # Small delay for visualization

    def auto_select_center_point(self):
        """Select center point from converged CVT diagram"""
        if len(self.points) == 0:
            return

        # Clear any previous selection
        self.fixed_point_index = None
        self.fixed_point_mode = True

        # Find geometric center of the canvas
        canvas_center = np.array([self.width / 2, self.height / 2])

        # Find which Voronoi region contains the center
        for i, region_index in enumerate(self.vor.point_region[:len(self.points)]):
            region = self.vor.regions[region_index]
            if -1 in region or len(region) == 0:
                continue

            polygon = Polygon(self.vor.vertices[region])
            if polygon.contains(Point(canvas_center)):
                self.fixed_point_index = i
                break

        # Fallback to closest point if no region contains center
        if self.fixed_point_index is None:
            points_array = np.array(self.points)
            distances = distance.cdist(points_array, [canvas_center])
            self.fixed_point_index = np.argmin(distances)

        # Update UI
        self.select_fixed_button.config(text="Select Point A (On)")
        self.highlight_center_point()

    def highlight_center_point(self):
        """Visual feedback for center point selection"""
        if self.fixed_point_index is not None:
            # Update plot
            self.plot_voronoi()

            # Add special marker
            x, y = self.points[self.fixed_point_index]
            self.ax.plot(x, y, 's', color='gold', markersize=12,
                         markeredgecolor='black', label='Center Point')
            self.ax.legend()

            if self.canvas:
                self.canvas.draw()

    def apply_radiant_optimization(self, levels=3, lambda_=0.5):
        centroids = self.calculate_centroids()
        level_polygons = self.get_neighborhood_polygons(self.fixed_point_index, levels)

        if level_polygons:
            level_centroids = [poly.centroid for level, poly in level_polygons]
            centroids_all = MultiPoint(level_centroids).centroid

            lloyd_pos = centroids[self.fixed_point_index]
            new_x = (1 - lambda_) * lloyd_pos[0] + lambda_ * centroids_all.x
            new_y = (1 - lambda_) * lloyd_pos[1] + lambda_ * centroids_all.y
            centroids[self.fixed_point_index] = (new_x, new_y)

            self.points = centroids
            self.vor = Voronoi(self.add_mirror_points(self.points))

            self.plot_voronoi()
            colors = ['#4CAF50', '#FF9800', '#9C27B0']
            if hasattr(self, 'neighborhood_polygons'):
                colors = ['#4CAF50', '#FF9800', '#9C27B0']
                for level, poly in self.neighborhood_polygons:
                    patch = PolygonPatch(poly,
                                         fc=colors[level - 1],
                                         ec=colors[level - 1],
                                         alpha=0.3,
                                         linewidth=1.5)
                    self.ax.add_patch(patch)

                    # Plot centroid
                    self.ax.plot(poly.centroid.x, poly.centroid.y, 'o',
                                 color=colors[level - 1], markersize=8)
            self.ax.plot(centroids_all.x, centroids_all.y, 'k*',
                         markersize=15, label='Radiant Center')
            self.ax.legend()
            self.canvas.draw()

    def get_voronoi_neighbors(self, idx):
        neighbors = set()
        for p1, p2 in self.vor.ridge_points:
            if p1 == idx and p2 < len(self.points):
                neighbors.add(p2)
            elif p2 == idx and p1 < len(self.points):
                neighbors.add(p1)
        return list(neighbors)

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

    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union

    def get_neighborhood_polygon(self, point_indices):
        """Create alpha shape polygon for neighborhood"""
        if len(point_indices) < 3:
            return None

        points = np.array([self.points[i] for i in point_indices])
        alpha = 0.5 * (1 + len(points) / 10)  # Dynamic alpha based on point density
        return alphashape.alphashape(points, alpha)

    def get_neighborhood_polygons(self, idx, max_levels):
        """Get exact neighborhood polygons for each level (concave)"""
        polygons = []
        all_previous = set()

        for level in range(1, max_levels + 1):
            # Get neighbors exactly at current level
            neighbor_indices = self.get_k_level_neighbors(idx, level)
            neighbor_indices = [i for i in neighbor_indices if i not in all_previous]

            if len(neighbor_indices) >= 3:  # Need at least 3 points for a polygon
                polygon = self.get_neighborhood_polygon(neighbor_indices)
                if polygon:
                    polygons.append((level, polygon))
                    all_previous.update(neighbor_indices)

        return polygons

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
    app = RadiantVoronoiGenerator(root)
    root.mainloop()