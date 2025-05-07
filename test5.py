import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import matplotlib.patches as mpatches


class VoronoiGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("CVT Generator with Lloyd's Algorithm")

        # Points storage
        self.points = []
        self.original_points = None  # Store original points for reference
        self.vor = None
        self.centroids = None
        self.iteration_count = 0
        self.show_original = False  # Flag to toggle original points display

        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas for clicking
        self.click_canvas = tk.Canvas(self.main_frame, bg='white', width=600, height=400)
        self.click_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create frame for buttons
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Add buttons
        self.clear_button = tk.Button(self.button_frame, text="Clear Points", command=self.clear_points)
        self.clear_button.pack(fill=tk.X, pady=5)

        self.generate_button = tk.Button(self.button_frame, text="Generate Voronoi", command=self.generate_voronoi)
        self.generate_button.pack(fill=tk.X, pady=5)

        self.lloyd_button = tk.Button(self.button_frame, text="Single Lloyd Step", command=self.apply_lloyd)
        self.lloyd_button.pack(fill=tk.X, pady=5)

        self.auto_lloyd_button = tk.Button(self.button_frame, text="Auto Lloyd to CVT", command=self.auto_lloyd)
        self.auto_lloyd_button.pack(fill=tk.X, pady=5)

        self.show_original_button = tk.Button(self.button_frame, text="Toggle Original Points",
                                              command=self.toggle_original_points)
        self.show_original_button.pack(fill=tk.X, pady=5)

        # Info label
        self.info_label = tk.Label(self.button_frame, text="Iterations: 0")
        self.info_label.pack(fill=tk.X, pady=5)

        # Bind click event
        self.click_canvas.bind("<Button-1>", self.add_point)

        # Matplotlib figure for Voronoi diagram
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = None

        # Set plot boundaries
        self.bounds = [0, 600, 0, 400]  # xmin, xmax, ymin, ymax

    def add_point(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))

        # Draw the point on canvas
        radius = 3
        self.click_canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='red')

    def clear_points(self):
        self.points = []
        self.original_points = None
        self.vor = None
        self.centroids = None
        self.iteration_count = 0
        self.show_original = False
        self.info_label.config(text="Iterations: 0")
        self.click_canvas.delete("all")
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.ax.clear()

    def generate_voronoi(self):
        if len(self.points) < 2:
            messagebox.showwarning("Warning", "You need at least 2 points to generate a Voronoi diagram")
            return

        # Convert points to numpy array
        points_array = np.array(self.points)
        self.original_points = points_array.copy()  # Store original points

        # Compute Voronoi diagram
        self.vor = Voronoi(points_array)
        self.centroids = None
        self.iteration_count = 0
        self.info_label.config(text="Iterations: 0")

        self.plot_voronoi()

    def plot_voronoi(self):
        # Clear previous plot
        self.ax.clear()

        # Plot bounded Voronoi diagram
        self.plot_bounded_voronoi()

        # Plot points and centroids
        if self.centroids is not None:
            # Draw arrows from old points to new centroids
            for i in range(len(self.vor.points)):
                dx = self.centroids[i, 0] - self.vor.points[i, 0]
                dy = self.centroids[i, 1] - self.vor.points[i, 1]
                self.ax.arrow(self.vor.points[i, 0], self.vor.points[i, 1],
                              dx, dy, head_width=5, head_length=5, fc='green', ec='green')

            self.ax.plot(self.vor.points[:, 0], self.vor.points[:, 1], 'ro')  # Current points
            self.ax.plot(self.centroids[:, 0], self.centroids[:, 1], 'go')  # Centroids
        else:
            self.ax.plot(self.vor.points[:, 0], self.vor.points[:, 1], 'ro')  # Only current points

        # Show original points if toggled
        if self.show_original and self.original_points is not None:
            self.ax.plot(self.original_points[:, 0], self.original_points[:, 1], 'bo', alpha=0.5)  # Original points

        self.ax.set_title(f'Voronoi Diagram (Iteration: {self.iteration_count})')
        self.ax.set_xlim(*self.bounds[:2])
        self.ax.set_ylim(*self.bounds[2:])

        # Remove old canvas if exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # Create new canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.click_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Hide the click canvas points
        self.click_canvas.delete("all")

    def plot_bounded_voronoi(self):
        """Plot Voronoi diagram properly bounded by the canvas dimensions"""
        # Create bounding box polygon
        bounding_box = Polygon([
            (self.bounds[0], self.bounds[2]),
            (self.bounds[1], self.bounds[2]),
            (self.bounds[1], self.bounds[3]),
            (self.bounds[0], self.bounds[3])
        ])

        # Plot finite ridges
        for simplex in self.vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):  # Finite ridge
                self.ax.plot(self.vor.vertices[simplex, 0], self.vor.vertices[simplex, 1], 'b-')

        # Plot infinite ridges properly clipped
        center = self.vor.points.mean(axis=0)
        for pointidx, simplex in zip(self.vor.ridge_points, self.vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):  # Infinite ridge
                i = simplex[simplex >= 0][0]  # Finite end
                p1, p2 = self.vor.points[pointidx] # Get the two points defining this ridge

                # Calculate direction vector
                t = p2 - p1
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # Normal vector

                # Find where the ridge crosses the bounding box
                midpoint = (p1 + p2) / 2
                direction = np.sign(np.dot(midpoint - center, n)) * n

                # Create a line from the finite end to far away in the direction
                far_point = self.vor.vertices[i] + direction * max(self.bounds[1], self.bounds[3]) * 2
                ridge_line = LineString([self.vor.vertices[i], far_point])

                # Clip with bounding box
                intersection = ridge_line.intersection(bounding_box)

                if not intersection.is_empty:
                    if intersection.geom_type == 'Point':
                        end_point = [intersection.x, intersection.y]
                    else:  # LineString (multiple intersections)
                        # Take the farthest point from the Voronoi vertex
                        coords = np.array(intersection.coords)
                        distances = np.linalg.norm(coords - self.vor.vertices[i], axis=1)
                        end_point = coords[np.argmax(distances)]

                    self.ax.plot([self.vor.vertices[i, 0], end_point[0]],
                                 [self.vor.vertices[i, 1], end_point[1]], 'b-')

    def apply_lloyd(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Please generate a Voronoi diagram first")
            return

        # Calculate centroids of Voronoi cells using Shapely
        self.centroids = self.calculate_centroids_with_shapely()

        # Create new Voronoi diagram with the centroids as points
        self.vor = Voronoi(self.centroids)
        self.iteration_count += 1
        self.info_label.config(text=f"Iterations: {self.iteration_count}")

        self.plot_voronoi()

    def auto_lloyd(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Please generate a Voronoi diagram first")
            return

        max_iterations = 100
        tolerance = 1e-6  # Convergence threshold
        prev_centroids = self.vor.points.copy()

        for _ in range(max_iterations):
            self.centroids = self.calculate_centroids_with_shapely()

            # Check for convergence
            movement = np.linalg.norm(self.centroids - prev_centroids, axis=1).max()
            if movement < tolerance:
                break

            prev_centroids = self.centroids.copy()
            self.vor = Voronoi(self.centroids)
            self.iteration_count += 1
            self.info_label.config(text=f"Iterations: {self.iteration_count}")
            self.root.update()  # Update the GUI to show progress

        self.plot_voronoi()

    def toggle_original_points(self):
        self.show_original = not self.show_original
        self.plot_voronoi()

    def calculate_centroids_with_shapely(self):
        centroids = np.zeros_like(self.vor.points)
        regions = [self.vor.regions[i] for i in self.vor.point_region]
        bounding_box = Polygon([
            (self.bounds[0], self.bounds[2]),
            (self.bounds[1], self.bounds[2]),
            (self.bounds[1], self.bounds[3]),
            (self.bounds[0], self.bounds[3])
        ])

        for i, region in enumerate(regions):
            if -1 in region or len(region) == 0:
                # Infinite or empty region - use original point as centroid
                centroids[i] = self.vor.points[i]
                continue

            # Get vertices of the Voronoi cell
            vertices = self.vor.vertices[region]

            # Create Shapely polygon
            poly = Polygon(vertices)

            # Clip with bounding box to handle infinite regions
            clipped_poly = poly.intersection(bounding_box)

            if clipped_poly.is_empty:
                centroids[i] = self.vor.points[i]
            else:
                # Calculate centroid using Shapely
                centroid = clipped_poly.centroid
                centroids[i] = [centroid.x, centroid.y]

        return centroids


if __name__ == "__main__":
    from shapely.geometry import LineString  # Import needed for bounded Voronoi

    root = tk.Tk()
    app = VoronoiGenerator(root)
    root.mainloop()