"""
This module is to define the plotting functions to be used
to trace the motion in the different experiments
"""

from libraries_module import plt

# Function to plot two dimensional data
def two_dimensional_plot(horizontal, vertical, title, horizontal_label, vertical_label, filename,
               line_plot = True):
    """
    Plots two-dimensional data using Matplotlib and saves it as a PNG file.

    Parameters:
    Inputs:
    horizontal (list or array): Data for the X-axis.
    vertical (list or array): Data for the Y-axis.
    title (str): Title of the plot.
    horizontal_label (str): Label for the X-axis.
    vertical_label (str): Label for the Y-axis.
    line_plot (bool): If True, creates a line plot; otherwise, a scatter plot.
    filename (str): Name of the file to save the plot as.
    """
    plt.figure(figsize=(10, 6))
    if line_plot:
        plt.plot(horizontal, vertical, marker='o')
    else:
        plt.scatter(horizontal, vertical)
    plt.title(title)
    plt.xlabel(horizontal_label)
    plt.ylabel(vertical_label)
    plt.grid(True)
    plt.savefig(filename, format = 'png')
    print(f"Plot saved as {filename}")
    plt.show()

# Functuion to show multiple plots on the same grid
def multi_plot(horizontal_list, vertical_list, title, horizontal_label, vertical_label, filename, legends,
               line_plot=True):
    """
    Plots multiple datasets on the same grid using Matplotlib and saves it as a PNG file.

    Parameters:
    Inputs:
    horizontal_list: A list of lists for the x-axis data for each plot.
    vertical_list: A list of lists for the y-axis data for each plot.
    title (str): Title of the plot.
    horizontal_label (str): Label for the X-axis.
    vertical_label (str): Label for the Y-axis.
    filename (str): Name of the file to save the plot as.
    legends (list of str): List of legend labels for each dataset in data lists.
    line_plot (bool): If True, creates line plots; otherwise, scatter plots.
    """
    plt.figure(figsize=(10, 6))
    lists_number = len(horizontal_list)
    # sizes = [len(i) for i in horizontal_list]
    for i in range(lists_number):
        if line_plot:
            plt.plot(horizontal_list[i], vertical_list[i], marker = 'o', label = legends[i] if legends else f"Dataset {i+1}")
        else:
            plt.scatter(horizontal_list[i], vertical_list[i], label = legends[i] if legends else f"Dataset {i+1}")
    plt.title(title)
    plt.xlabel(horizontal_label)
    plt.ylabel(vertical_label)
    plt.grid(True)
    if legends:
        plt.legend()
    plt.savefig(filename, format = 'png')
    print(f"Plot saved as {filename}")
    plt.show()
