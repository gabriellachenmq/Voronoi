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

        self.fixed_point_mode = False
        self.fixed_point_index = None

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Change canvas dimensions to square (500x500)
        self.canvas_size = 600
        self.click_canvas = tk.Canvas(self.main_frame, bg='white', width=self.canvas_size, height=self.canvas_size)
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

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = None

        self.bounds = [0, self.canvas_size, 0, self.canvas_size]
        self.width = self.bounds[1] - self.bounds[0]
        self.height = self.bounds[3] - self.bounds[2]

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

        return np.vstack([points] + mirrored)

    def generate_voronoi(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "Need at least 2 points")
            return

        self.points = np.array(self.points)
        self.vor = Voronoi(self.add_mirror_points(self.points))
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

        # Plot all points
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
        for _ in range(10):
            self.apply_lloyd()

        if self.auto_previous_points is not None:
            movement = np.linalg.norm(self.auto_previous_points - self.points)
            if movement < self.auto_tolerance or self.iteration_count == 150:
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

    def is_boundary_point(self, point_idx):

        if point_idx >= len(self.points):  # Handle mirror points
            return True

        region_index = self.vor.point_region[point_idx]
        if region_index == -1:  # Invalid region
            return True

        region = self.vor.regions[region_index]
        if not region:  # Empty region
            return True

        for vertex_idx in region:
            if vertex_idx == -1:  # Infinite vertex
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

        return max_levels if max_levels > 0 else 1  # Return at least 1 level

    def apply_radiant_algorithm(self):
        if self.fixed_point_index is None:
            messagebox.showwarning("Warning", "No central point. Run CVT first.")
            return

        if self.vor is None:
            messagebox.showwarning("Warning", "Generate Voronoi first.")
            return

        # Calculate max_levels dynamically
        max_levels = self.calculate_max_levels(self.fixed_point_index)
        level_polygons = self.get_neighborhood_polygons(self.fixed_point_index, max_levels+1)

        if not level_polygons:
            return

        # Rest of your radiant algorithm implementation...
        centroids = self.calculate_centroids()
        lloyd_centroid = centroids[self.fixed_point_index]

        level_centroids = [poly.centroid for level, poly in level_polygons]
        centroids_all = MultiPoint(level_centroids).centroid

        lambda_ = 0.5  # Adjust this for different blending
        new_x = (1 - lambda_) * lloyd_centroid[0] + lambda_ * centroids_all.x
        new_y = (1 - lambda_) * lloyd_centroid[1] + lambda_ * centroids_all.y
        centroids[self.fixed_point_index] = (new_x, new_y)

        self.auto_previous_points = np.array(self.points)
        self.points = centroids
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

        safe_levels_exist = False
        for level, poly in level_polygons:
            neighbor_indices = self.get_k_level_neighbors(self.fixed_point_index, level)
            if not any(self.is_boundary_point(i) for i in neighbor_indices):
                safe_levels_exist = True
                break

        max_level = max(level for level, _ in level_polygons) if level_polygons else 0 # 3
        min_level = min(level for level, _ in level_polygons) if level_polygons else 0

        for level, poly in level_polygons:
            neighbor_indices = self.get_k_level_neighbors(self.fixed_point_index, level)

            if level == max_level and not any(self.is_boundary_point(i) for i in neighbor_indices):
                self.ax.plot(*poly.exterior.xy, color='black',
                             linewidth=2, linestyle='--',
                             label=f'Level {level-1} neighborhood')
            elif level == min_level:
                self.ax.plot(*poly.exterior.xy, color='green',
                             linewidth=2, linestyle='--',
                             label=f'Level {level-1} neighborhood')
            else:
                self.ax.plot(*poly.exterior.xy, color='black',
                             linewidth=1, linestyle=':',
                             alpha=0.5)

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

    def progressive_radiant_central_outward(self, tol=1e-6, max_iters=200, lam=0.5):
        """
        Sweep from central point outward:
        1. For each level k (1...max_level), do Lloyd/Radiant Lloyd for ONLY A (the central point, updated as you go)
           and the points in k-level, fixing all inner levels' points.
        2. Stop a level when all those points (central + level-k neighbors) converge (< tol).
        3. After all levels, if any point moved more than tol, repeat from k=1.
        4. Stop when one sweep from 1 to max_level makes all levels converged (< tol movement).
        """

        if self.fixed_point_index is None:
            messagebox.showinfo("Info", "Run CVT to determine central point first.")
            return

        npts = len(self.points)
        max_levels = self.calculate_max_levels(self.fixed_point_index)
        self.iteration_count = 0

        converged = False

        # Track fixed points for visualization (will only fix after each level)
        self.fixed_points = set()

        while not converged:
            converged = True  # If any sweep changes points more than tol, set to False and repeat
            central_idx = self.fixed_point_index
            central_pos = self.points[central_idx]  # Always take most up-to-date A position

            for level in range(1, max_levels + 1):
                # Gather all already-fixed indices
                fixed_indices = set()
                for lev_inner in range(1, level):  # fixed all inner levels
                    fixed_indices.update(self.get_k_level_neighbors(central_idx, lev_inner))

                # Points to "relax" this round: A and k-level points (not previously fixed or mirrors):
                level_indices = set(self.get_k_level_neighbors(central_idx, level))
                relax_indices = level_indices - fixed_indices

                # Always include central point if not in fixed:
                if central_idx not in fixed_indices:
                    relax_indices.add(central_idx)

                # Skip if none unfixed (can happen on boundaries)
                if not relax_indices:
                    continue

                # Prepare to check convergence at this level
                movement_this_level = float('inf')
                inner_iters = 0

                while movement_this_level > tol and inner_iters < max_iters:
                    inner_iters += 1
                    old_positions = {i: np.array(self.points[i]) for i in relax_indices}

                    centroids = self.calculate_centroids()
                    # Lloyd for all level points (except central)
                    for idx in relax_indices:
                        if idx == central_idx:
                            continue
                        self.points[idx] = tuple(centroids[idx])

                    # Radiant Lloyd for the central point with ONLY this level's polygon
                    if central_idx in relax_indices:
                        level_polygons = self.get_neighborhood_polygons(central_idx, level)
                        poly = None
                        for lvl, p in level_polygons:
                            if lvl == level:
                                poly = p
                                break
                        if poly is not None and not poly.is_empty:
                            radiant_centroid = poly.centroid
                            lloyd_centroid = centroids[central_idx]
                            new_x = (1 - lam) * lloyd_centroid[0] + lam * radiant_centroid.x
                            new_y = (1 - lam) * lloyd_centroid[1] + lam * radiant_centroid.y
                            self.points[central_idx] = (new_x, new_y)

                    # Recompute Voronoi for next iteration
                    self.vor = Voronoi(self.add_mirror_points(self.points))

                    # Check the maximum movement among relaxed points
                    new_positions = {i: np.array(self.points[i]) for i in relax_indices}
                    movement_this_level = max(
                        np.linalg.norm(new_positions[i] - old_positions[i]) for i in relax_indices)

                    self.iteration_count += 1
                    # Optional: plot each sub-iteration
                    level_polygons = self.get_neighborhood_polygons(central_idx, level)
                    self.plot_radiant_voronoi(level_polygons)
                    self.root.update_idletasks()

                # After finishing this level, mark these points as fixed for this sweep
                self.fixed_points.update(relax_indices)

                # If there was significant movement, this sweep did some work, so plan to repeat
                if movement_this_level > tol:
                    converged = False

            # At end of one 1-to-max sweep, clear fixed_points so we allow all to update next time
            self.fixed_points = set()  # Visualization only
            # Or, if you want fixed points to persist, modify logic accordingly

        self.info_label.config(text=f"Sweeps Iter: {self.iteration_count}")
        messagebox.showinfo("Info", f"Progressive radiant CVT finished in {self.iteration_count} iterations.")
        # Final plot
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