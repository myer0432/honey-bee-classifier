from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('classic')

# @param x: A range(n) of numbers for the x axis
# @param y: A list containing lists containing the values for each line to be plotted
# @param ylabels: A list containing the labels to the lines in y in the respective order
# @param xaxis: The label for the x-axis
# @param yaxis: The label for the y-axis
# @param title: The title for the plot
# @param subtitle: The subtitle for the plot (optional)
class plot:
    # Constructor
    def __init__(self, x, y, ylabels, xaxis, yaxis, title, subtitle=None):
        if subtitle:
            plt.axes([.1,.1,.8,.7])
            plt.figtext(.5,.9,title, fontsize=20, ha="center")
            plt.figtext(.5,.85,subtitle,fontsize=18,ha="center")
        else:
            plt.title(title)
        plt.xlabel(xaxis, fontsize=18)
        plt.ylabel(yaxis, fontsize=18)
        default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                  cycler(linestyle=['-', '--', ':', '-.']))
        plt.rc('lines', linewidth=2)
        plt.rc('axes', prop_cycle=default_cycler)
        for i in range(len(y)):
            plt.plot(x, y[i], label=ylabels[i])
        plt.legend(loc="lower right")

    # Show figure
    def show(self):
        plt.show()
