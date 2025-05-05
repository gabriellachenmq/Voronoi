import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class VoronoiGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Voronoi Diagram Generator with Lloyd's Algorithm")

        # Points storage
        self.points = []
        self.vor = None
        self.centroids = None

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

        self.lloyd_button = tk.Button(self.button_frame, text="Lloyd's Algorithm", command=self.apply_lloyd)
        self.lloyd_button.pack(fill=tk.X, pady=5)

        # Bind click event
        self.click_canvas.bind("<Button-1>", self.add_point)

        # Matplotlib figure for Voronoi diagram
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = None

    def add_point(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))

        # Draw the point on canvas
        radius = 3
        self.click_canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='red')

    def clear_points(self):
        self.points = []
        self.vor = None
        self.centroids = None
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

        # Compute Voronoi diagram
        self.vor = Voronoi(points_array)
        self.centroids = None

        self.plot_voronoi()

    def plot_voronoi(self):
        # Clear previous plot
        self.ax.clear()

        # Plot Voronoi diagram
        voronoi_plot_2d(self.vor, ax=self.ax, show_vertices=False, line_colors='blue',
                        line_width=1, line_alpha=0.6, point_size=5)

        # Plot original points in red and centroids in green if available
        if self.centroids is not None:
            self.ax.plot(self.vor.points[:, 0], self.vor.points[:, 1], 'ro')  # Original points
            self.ax.plot(self.centroids[:, 0], self.centroids[:, 1], 'go')  # Centroids
        else:
            self.ax.plot(self.vor.points[:, 0], self.vor.points[:, 1], 'ro')  # Only original points

        self.ax.set_title('Voronoi Diagram')
        self.ax.set_xlim(0, 600)
        self.ax.set_ylim(0, 400)

        # Remove old canvas if exists
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # Create new canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.click_canvas)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Hide the click canvas points
        self.click_canvas.delete("all")

    def apply_lloyd(self):
        if self.vor is None:
            messagebox.showwarning("Warning", "Please generate a Voronoi diagram first")
            return

        # Calculate centroids of Voronoi cells
        self.centroids = self.calculate_centroids()

        # Create new Voronoi diagram with the centroids as points
        self.vor = Voronoi(self.centroids)

        self.plot_voronoi()

    def calculate_centroids(self):
        centroids = np.zeros_like(self.vor.points)
        regions = [self.vor.regions[i] for i in self.vor.point_region]

        for i, region in enumerate(regions):
            if -1 in region or len(region) == 0:
                # Infinite region or empty region - use original point as centroid
                centroids[i] = self.vor.points[i]
                continue

            # Get vertices of the Voronoi cell
            vertices = self.vor.vertices[region]

            # Calculate centroid (geometric center) of the polygon
            polygon = Polygon(vertices, closed=True)
            centroids[i] = polygon.get_xy().mean(axis=0)

        return centroids


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiGenerator(root)
    root.mainloop()