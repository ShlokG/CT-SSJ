# This file includes the function used for plotting the runtimes.
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import FuncFormatter
import numpy as np

# Plot the runtimes
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rc('legend',fontsize=16)

# Define a formatter function
def thousands_formatter(x, pos):
    return f'{int(x/1000)}'

def plot_fn(df, str_append = "", no_legend = False, fig_dir = "Figures/"):
    """Plot the runtimes for the steady state for each of the three methods.
    
    Args:
        df (pd.DataFrame): DataFrame containing the runtimes for each method
        str_append (str): String to append to the filename
        no_legend (bool): Whether to include a legend in the plot
        fig_dir (str): Directory to save the figure to

    Returns:
        None (just saves the figure to specified file)
    """

    yval = 'Steady State'
    ymax = math.ceil(df[yval].max())

    # Plot the data
    df1 = df[df['Type'] == 'Continuous, EGM']
    df2 = df[df['Type'] == 'Continuous, Implicit']
    df3 = df[df['Type'] == 'Discrete']
    plt.plot(df1['Gridpoints'], df1[yval], label='Continuous, EGM', linestyle='--', linewidth=3, color=default_colors[2])
    plt.plot(df2['Gridpoints'], df2[yval], label='Continuous, Implicit', linestyle=':', linewidth=3, color=default_colors[3])
    plt.plot(df3['Gridpoints'], df3[yval], label='Discrete', linestyle='-.', linewidth=3, color=default_colors[1])

    min_df = df[df['Type'] != 'Discrete'].groupby('Gridpoints', as_index=False)[yval].min()
    plt.plot(min_df['Gridpoints'], min_df[yval], label='Continuous, Minimum', linestyle='-', \
                linewidth=3, color = default_colors[0])

    if not no_legend:
        plt.legend(frameon=False)

    # Add labels and title
    plt.xlabel('Gridpoints (thousands)', fontsize=14)
    plt.ylabel('Runtime (seconds)', fontsize=14)

    # Increase tick label sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim(0, ymax)
    plt.xlim(0, df['Gridpoints'].max())

    plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.savefig(fig_dir + "runs" + str_append + ".pdf")
    plt.show()

def plot_cumulative_runtime(df, typed = "Continuous", str_append="", no_legend = False, fig_dir = "Figures/"):
    """Plot the runtimes for each step of the Jacobian calculation for given method.
    
    Args:
        df (pd.DataFrame): DataFrame containing the runtimes for each method
        typed (str): Type of solution method to focus on
        str_append (str): String to append to the filename
        no_legend (bool): Whether to include a legend in the plot
        fig_dir (str): Directory to save the figure to

    Returns:
        None (just saves the figure to specified file)
    """

    # Get the list of columns starting from 'Step 1'
    step_columns = [col for col in df.iloc[:,3:8]]

    # Initialize cumulative sum DataFrame
    cumulative_df = df[['Gridpoints', 'Type']].copy()
    cumulative_df[step_columns[0]] = df[step_columns[0]]

    # Calculate cumulative sums
    for i in range(1, len(step_columns)):
        cumulative_df[step_columns[i]] = cumulative_df[step_columns[i-1]] + df[step_columns[i]]

    ymax = math.ceil(cumulative_df.iloc[:, -1].max())
    
    # Focus on given type
    cumulative_df = cumulative_df[cumulative_df['Type'] == typed]

    # Plot cumulative runtime for each step in a single plot
    previous_step_data = np.zeros_like(cumulative_df['Gridpoints'])
    plt.figure(figsize=(10, 6))
    linestyles = ['solid', 'solid', 'solid', 'dashed', 'dotted']
    for step_num in range(len(step_columns)):
        step = step_columns[step_num]
        plt.plot(cumulative_df['Gridpoints'], cumulative_df[step],
                 label=step, linewidth=3, linestyle = linestyles[step_num])
        plt.fill_between(cumulative_df['Gridpoints'], previous_step_data, cumulative_df[step], alpha=0.3)
        previous_step_data = cumulative_df[step]

    # Add legend
    if not no_legend:
        plt.legend(frameon=False)

    # Add labels and title
    plt.xlabel('Gridpoints (thousands)', fontsize=16)
    plt.ylabel('Cumulative Runtime (seconds)', fontsize=16)

    # Increase tick label sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Apply the formatter to the x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.ylim(0, ymax)
    plt.savefig(fig_dir + "cumulative_runtime" + str_append + ".pdf", dpi = 1200)
    plt.show()
