import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
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

        self.selected_point_A_index = None  # Index of the fixed special point A
        self.select_point_A_mode = False    # Are we in "Select Point A" mode?

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

        # New Button to toggle select point A mode
        self.select_point_A_button = tk.Button(self.button_frame, text="Select Point A", command=self.toggle_select_point_A)
        self.select_point_A_button.pack(fill=tk.X, pady=5)

        self.info_label = tk.Label(self.button_frame, text="Iterations: 0")
        self.info_label.pack(fill=tk.X, pady=5)

        self.click_canvas.bind("<Button-1>", self.on_canvas_click)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = None

        self.bounds = [0, 600, 0, 400]
        self.width = self.bounds[1] - self.bounds[0]
        self.height = self.bounds[3] - self.bounds[2]

    def toggle_select_point_A(self):
        self.select_point_A_mode = not self.select_point_A_mode
        if self.select_point_A_mode:
            self.select_point_A_button.config(relief=tk.SUNKEN)
            messagebox.showinfo("Select Point A", "Click near a point to select it as Point A (green).")
        else:
            self.select_point_A_button.config(relief=tk.RAISED)

    def on_canvas_click(self, event):
        if self.select_point_A_mode:
            # Try to select a point near the click as Point A
            click_x, click_y = event.x, self.height - event.y
            if len(self.points) == 0:
                messagebox.showwarning("Warning", "No points to select.")
                return

            points_arr = np.array(self.points)
            distances = np.linalg.norm(points_arr - np.array([click_x, click_y]), axis=1)
            closest_index = np.argmin(distances)
            if distances[closest_index] < 15:  # pixel threshold for selecting a point
                self.selected_point_A_index = closest_index
                self.select_point_A_mode = False
                self.select_point_A_button.config(relief=tk.RAISED)
                self.plot_voronoi()
                messagebox.showinfo("Point A Selected", f"Selected point {closest_index} as Point A.")
            else:
                messagebox.showwarning("Warning", "Click closer to an existing point to select Point A.")
        else:
            # Normal adding point mode
            self.add_point(event)

    def add_point(self, event):
        x, y = event.x, self.height - event.y  # Convert y-coordinate here
        self.points.append((x, y))
        self.original_points.append((x, y))
        self.click_canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill='red')

    # ... (keep your existing methods unchanged here) ...

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

        pts = np.array(self.points)
        # Draw all points in red
        self.ax.plot(pts[:, 0], pts[:, 1], 'ro', label='Current Points')

        # Highlight point A in green if selected
        if self.selected_point_A_index is not None and 0 <= self.selected_point_A_index < len(self.points):
            a_pt = pts[self.selected_point_A_index]
            self.ax.plot(a_pt[0], a_pt[1], 'go', markersize=10, label='Point A')

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

        centroids = self.calculate_centroids()
        centroids = np.array(centroids)

        # If Point A is selected
        if self.selected_point_A_index is not None:
            A_idx = self.selected_point_A_index

            neighbors = self.get_voronoi_neighbors(A_idx)

            if len(neighbors) > 0:
                neighbor_points = [self.points[i] for i in neighbors]

                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(neighbor_points)
                    polygon_points = neighbor_points[hull.vertices]
                except Exception:
                    polygon_points = neighbor_points

                polygon = Polygon(polygon_points)
                if not polygon.is_valid or polygon.is_empty:
                    centroid_P = centroids[A_idx]
                else:
                    centroid_P = np.array([polygon.centroid.x, polygon.centroid.y])

                # Soft constraint update for point A with lambda=0.5
                lambda_ = 0.5
                centroids[A_idx] = (1 - lambda_) * centroids[A_idx] + lambda_ * centroid_P

        all_points = self.add_mirror_points(centroids)
        self.vor = Voronoi(all_points)
        self.points = centroids.copy()
        self.iteration_count += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")
        self.plot_voronoi()

    def get_voronoi_neighbors(self, point_index):
        neighbors = set()
        ridge_points = self.vor.ridge_points
        for (p1, p2) in ridge_points:
            if p1 == point_index:
                neighbors.add(p2)
            elif p2 == point_index:
                neighbors.add(p1)
        return list(neighbors)

    # The rest of your unchanged methods below...

    def clear_points(self):
        self.points = []
        self.original_points = []
        self.vor = None
        self.original_vor = None
        self.centroids = None
        self.iteration_count = 0
        self.auto_running = False
        self.selected_point_A_index = None
        self.select_point_A_mode = False
        self.select_point_A_button.config(relief=tk.RAISED)
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

        # left boundary (x=0)
        left_mirror = np.copy(points)
        left_mirror[:, 0] = -left_mirror[:, 0]
        mirrored.append(left_mirror)

        # right boundary (x=width)
        right_mirror = np.copy(points)
        right_mirror[:, 0] = 2 * self.width - right_mirror[:, 0]
        mirrored.append(right_mirror)

        # bottom boundary (y=0)
        bottom_mirror = np.copy(points)
        bottom_mirror[:, 1] = -bottom_mirror[:, 1]
        mirrored.append(bottom_mirror)

        # top boundary (y=height)
        top_mirror = np.copy(points)
        top_mirror[:, 1] = 2 * self.height - top_mirror[:, 1]
        mirrored.append(top_mirror)

        # Combine original + mirrors
        all_points = np.vstack([points] + mirrored)
        return all_points

    def calculate_centroids(self):
        centroids = []
        bounding_box = Polygon([
            (self.bounds[0], self.bounds[2]),
            (self.bounds[1], self.bounds[2]),
            (self.bounds[1], self.bounds[3]),
            (self.bounds[0], self.bounds[3])
        ])
        for i, region_index in enumerate(self.vor.point_region[:len(self.points)]):
            region = self.vor.regions[region_index]
            if -1 in region or len(region) == 0:
                centroids.append(self.points[i])  # fallback: keep old position
                continue
            polygon = Polygon(self.vor.vertices[region])
            clipped = polygon.intersection(bounding_box)
            if clipped.is_empty:
                centroids.append(self.points[i])  # fallback
            else:
                centroids.append(np.array([clipped.centroid.x, clipped.centroid.y]))
        return centroids

    def generate_voronoi(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "Add at least two points to generate Voronoi diagram.")
            return
        points_arr = np.array(self.points)
        all_points = self.add_mirror_points(points_arr)
        self.vor = Voronoi(all_points)
        self.iteration_count = 0
        self.info_label.config(text="Iterations: 0")
        self.plot_voronoi()

    def show_original_points(self):
        if len(self.original_points) == 0:
            messagebox.showwarning("Warning", "No original points to show.")
            return
        self.plot_voronoi(show_original=True)

    def auto_lloyd(self):
        if self.auto_running:
            self.auto_running = False
            self.auto_lloyd_button.config(text="Auto CVT")
        else:
            if self.vor is None:
                messagebox.showwarning("Warning", "Generate Voronoi diagram first.")
                return
            self.auto_running = True
            self.auto_lloyd_button.config(text="Stop Auto CVT")
            self.run_auto_lloyd()

    def run_auto_lloyd(self):
        if not self.auto_running:
            return
        self.apply_lloyd()
        self.root.after(500, self.run_auto_lloyd)


def main():
    root = tk.Tk()
    app = VoronoiGenerator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
