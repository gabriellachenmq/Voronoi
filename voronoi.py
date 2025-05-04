import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class VoronoiGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Voronoi Diagram Generator")

        # Points storage
        self.points = []

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
        vor = Voronoi(points_array)

        # Clear previous plot
        self.ax.clear()

        # Plot Voronoi diagram
        voronoi_plot_2d(vor, ax=self.ax, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.6,
                        point_size=5)
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


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiGenerator(root)
    root.mainloop()