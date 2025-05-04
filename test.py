import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import Voronoi, voronoi_plot_2d
import tkinter as tk
from tkinter import ttk


class CVTVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("CVT Visualizer")

        # Initialize points
        self.points = []
        self.voronoi = None
        self.regions = []
        self.vertices = []

        # Create GUI elements
        self.create_widgets()

        # Set up matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.set_title("Click to place points, then click 'Generate Voronoi'")

        # Embed matplotlib figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def create_widgets(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.generate_btn = ttk.Button(
            control_frame,
            text="Generate Voronoi",
            command=self.generate_voronoi
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.lloyd_btn = ttk.Button(
            control_frame,
            text="Lloyd's Algorithm (1 step)",
            command=self.lloyd_step,
            state=tk.DISABLED
        )
        self.lloyd_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.auto_btn = ttk.Button(
            control_frame,
            text="Auto Converge (10 steps)",
            command=self.auto_converge,
            state=tk.DISABLED
        )
        self.auto_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_btn = ttk.Button(
            control_frame,
            text="Clear All",
            command=self.clear_all
        )
        self.clear_btn.pack(side=tk.RIGHT, padx=5, pady=5)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Add point to our list
        self.points.append([event.xdata, event.ydata])

        # Redraw points
        self.redraw_points()

    def redraw_points(self):
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')

        if len(self.points) > 0:
            points_array = np.array(self.points)
            self.ax.scatter(points_array[:, 0], points_array[:, 1], c='red')

        if self.voronoi is not None:
            voronoi_plot_2d(self.voronoi, ax=self.ax, show_vertices=False)

        self.ax.set_title(f"{len(self.points)} points placed")
        self.canvas.draw()

    def generate_voronoi(self):
        if len(self.points) < 2:
            return

        points_array = np.array(self.points)
        self.voronoi = Voronoi(points_array)

        # Enable Lloyd's algorithm button
        self.lloyd_btn.config(state=tk.NORMAL)
        self.auto_btn.config(state=tk.NORMAL)

        self.redraw_points()

    def lloyd_step(self):
        if self.voronoi is None or len(self.points) < 2:
            return

        # Calculate centroids of Voronoi regions
        new_points = []
        points_array = np.array(self.points)

        for i in range(len(self.points)):
            region = self.voronoi.regions[self.voronoi.point_region[i]]
            if -1 not in region and len(region) > 0:
                polygon = self.voronoi.vertices[region]
                centroid = self.polygon_centroid(polygon)
                new_points.append(centroid)
            else:
                # Keep the original point if the region is invalid
                new_points.append(self.points[i])

        self.points = new_points
        self.voronoi = Voronoi(np.array(self.points))
        self.redraw_points()

    def auto_converge(self, steps=10):
        for _ in range(steps):
            self.lloyd_step()
            self.root.update()  # Update the GUI between steps

    def polygon_centroid(self, vertices):
        """Calculate the centroid of a polygon."""
        x = vertices[:, 0]
        y = vertices[:, 1]
        area = 0
        centroid_x = 0
        centroid_y = 0

        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            cross = (x[i] * y[j]) - (x[j] * y[i])
            area += cross
            centroid_x += (x[i] + x[j]) * cross
            centroid_y += (y[i] + y[j]) * cross

        area *= 0.5
        if area == 0:
            return vertices.mean(axis=0)

        centroid_x /= (6 * area)
        centroid_y /= (6 * area)

        return [centroid_x, centroid_y]

    def clear_all(self):
        self.points = []
        self.voronoi = None
        self.lloyd_btn.config(state=tk.DISABLED)
        self.auto_btn.config(state=tk.DISABLED)
        self.redraw_points()


if __name__ == "__main__":
    root = tk.Tk()
    app = CVTVisualizer(root)
    root.mainloop()