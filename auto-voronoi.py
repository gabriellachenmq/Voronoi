import tkinter as tk
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class VoronoiApp:
    def __init__(self, master, min_points=3):  # Changed minimum to 3 (Voronoi needs at least 3 points)
        self.master = master
        self.master.title("Voronoi Diagram Generator")
        self.min_points = min_points
        self.points = []

        # Create a matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind mouse click
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Set up plot limits (you can adjust these as needed)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        # Initial empty plot
        self.update_diagram()

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks outside plot area

        self.points.append([event.xdata, event.ydata])
        self.update_diagram()

    def update_diagram(self):
        self.ax.clear()

        # Always plot the points
        if self.points:
            x, y = zip(*self.points)
            self.ax.plot(x, y, 'ko')

        # Generate Voronoi diagram if we have enough points
        if len(self.points) >= self.min_points:
            try:
                vor = Voronoi(self.points)
                voronoi_plot_2d(vor, ax=self.ax, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.6)
            except:
                # Handle cases where Voronoi diagram can't be computed (e.g., colinear points)
                pass

        # Redraw the points on top of the Voronoi diagram
        if self.points:
            x, y = zip(*self.points)
            self.ax.plot(x, y, 'ko', markersize=5)

        # Set plot properties
        self.ax.set_title(f"Points: {len(self.points)} (Click to add more)")
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_aspect('equal')
        # self.ax.grid(True)

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiApp(root)
    root.mainloop()