import ternary
import matplotlib
fig, tax = ternary.figure(scale=100)
fig.set_size_inches(10, 9)

# Plot points.
points =  [(0,0,2), (0,0,2), (0,0,2)]
points = [(y, z, x) for (x, y, z) in points]
tax.plot_colored_trajectory(points, cmap="hsv", linewidth=2.0)

# Axis labels. (See below for corner labels.)
fontsize = 14
offset = 0.08
tax.left_axis_label("Rock %", fontsize=fontsize, offset=offset)
tax.right_axis_label("Paper %", fontsize=fontsize, offset=offset)
tax.bottom_axis_label("Scissor %", fontsize=fontsize, offset=-offset)
tax.set_title("Sobrarbe Formation", fontsize=20)

# Decoration.
tax.boundary(linewidth=1)
tax.gridlines(multiple=10, color="gray")
tax.ticks(axis='lbr', linewidth=1, multiple=20)
tax.get_axes().axis('off')

tax.show()