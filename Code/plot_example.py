#################################
# Example of how to use plot.py #
#################################
from plot import plot

x = range(5)
y = [ [0, 1, 2, 3, 4],
      [4, 3, 2, 1, 0],
      [2, 2, 2, 2, 2],
      [2, 3, 4, 3, 2] ]
ylabels = ["Line 1", "Line 2", "Line 3", "Line 4"]

# constructor signature:
# plot(self, x, y, ylabels, xaxis, yaxis, title, subtitle=None):
plot = plot(x, y, ylabels, "Index", "Number", "Index vs. Number", "Optional Subtitle")
plot.show()
