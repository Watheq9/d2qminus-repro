
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import scienceplots



def compare_two_histograms(array1, array2, name1, name2, save_file, 
                        x_label='Relevance Score', y_label='Frequency', 
                        colors=['#f58633','#3077b0',], title=None,
                        pdf_x_dim = 6.0, pdf_y_dim = 4.0, FS = 12):
    '''
    data: list of dictionaries, each dictionary has the following keys {x_values, y_values , label, color, marker}
    '''
    # Set font size
    params = {'axes.labelsize': FS,'axes.titlesize': FS, 'legend.fontsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS}
    plt.rcParams.update(params)
    # Plot histograms for both lists in a single figure

    # Subplots can be used to put multiple plots in one frame
    fig, ax = plt.subplots(figsize=(pdf_x_dim, pdf_y_dim))

    # Remove the top and right borders on the box around the figure
    ax.spines[['right', 'top']].set_visible(False)

    # Also make sure there are no ticks on the top or right
    ax.tick_params(which="both", top=False, right=False)

    plt.hist([array1, array2], bins=np.arange(min(min(array1), min(array2)), 
                                            max(max(array1), max(array2)) + 1.5) - 0.5, 
                            edgecolor='k', label=[name1, name2], color=colors,)
    
    plt.grid(axis='y', color='0.8', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='upper left')

    plt.savefig(save_file, dpi=300)



def draw_histogram(array, name, save_file, x_label='Relevance Score', y_label='Frequency'):
    # Plot histograms for both lists in a single figure
    plt.hist([array,], bins=np.arange(min(array), max(array) + 1.5) - 0.5, 
                                            edgecolor='k', label=[name,])
    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title('Histograms for Two Lists')

    # Add a legend
    plt.legend()
    plt.savefig(save_file, dpi=300)
    # Show the plot
    plt.show()


def draw_line_graph(data, save_file, x_axis_label=None, y_axis_label=None, title=None,
                    pdf_x_dim = 6.0, pdf_y_dim = 4.0, FS = 12, vertical_line=None):
    '''
    data: list of dictionaries, each dictionary has the following keys {x_values, y_values , label, color, marker}
    '''

    # Set font size
    params = {'axes.labelsize': FS,'axes.titlesize': FS, 'legend.fontsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS}
    plt.rcParams.update(params)
    # Subplots can be used to put multiple plots in one frame
    fig, ax = plt.subplots(figsize=(pdf_x_dim, pdf_y_dim))

    # Remove the top and right borders on the box around the figure
    ax.spines[['right', 'top']].set_visible(False)

    # Also make sure there are no ticks on the top or right
    ax.tick_params(which="both", top=False, right=False)

    for my_dict in data:
        ax.plot(my_dict['x_values'], my_dict['y_values'], label=my_dict['label'], color=my_dict['color'], marker=my_dict['marker'],)
        # plt.plot(my_dict['x_values'], my_dict['y_values'], label=my_dict['label'], color=my_dict['color'], marker=my_dict['marker'],)


    # draw vertical lines if any
    if vertical_line is not None:
        for line_properties in vertical_line:
            plt.axvline(x=line_properties['x_index'], color=line_properties['color'], linestyle=line_properties['linestyle'], label=line_properties['label'])

    # Set axes and legend
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend(loc='upper left')

    # Add a grid on top of the ticks for improved readibility
    plt.grid(axis='x', color='0.8', linestyle='--')
    plt.grid(axis='y', color='0.8', linestyle='--')

    # Output as a PDF
    plt.savefig(save_file, dpi=300)



def draw_boxplot(data, labels, save_file, x_axis_label=None, y_axis_label=None, title=None,
                    pdf_x_dim = 6.0, pdf_y_dim = 4.0, FS = 10, y_min_value=None, y_max_value=None):
    '''
    data: list of lists, each list will draw a boxplot
    labels: the names of each list to be drawn in the x-axis
    
    '''
    # Set font size
    params = {'axes.labelsize': FS,'axes.titlesize': FS, 'legend.fontsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS}
    plt.rcParams.update(params)
    # Subplots can be used to put multiple plots in one frame
    fig, ax = plt.subplots(figsize=(pdf_x_dim, pdf_y_dim))

    # Remove the top and right borders on the box around the figure
    ax.spines[['right', 'top']].set_visible(False)

    # Also make sure there are no ticks on the top or right
    ax.tick_params(which="both", top=False, right=False)

    plt.boxplot(data, labels=labels)

    ax.set_ylim(y_min_value, y_max_value)

    # Set axes and legend
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    # plt.legend(loc='upper left')

    plt.savefig(save_file, dpi=300)



def compare_two_stacked_charts(categories, x_ticks, model1_data, model1_label, model1_style,
                         model2_data, model2_label, model2_style,
                         save_file, x_axis_label=None, y_axis_label=None, title=None,
                         category_colors = ['skyblue', 'lightcoral'],
                    pdf_x_dim = 6.0, pdf_y_dim = 4.0, FS = 10, width=0.7, edgecolor='black'):

    params = {'axes.labelsize': FS,'axes.titlesize': FS, 'legend.fontsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS}

    plt.rcParams.update(params)
    # Subplots can be used to put multiple plots in one frame
    fig, ax = plt.subplots(figsize=(pdf_x_dim, pdf_y_dim))
    # Remove the top and right borders on the box around the figure
    ax.spines[['right', 'top']].set_visible(False)
    # Also make sure there are no ticks on the top or right
    ax.tick_params(which="both", top=False, right=False)

    x = np.arange(0, len(x_ticks) * 2, 2)
    # Plot the first stacked columns
    bottom = np.zeros(len(x_ticks))
    # Normalize percentages to ensure they add up to 100 for each model
    percentages_normalized = model1_data / np.array(model1_data).sum(axis=1, keepdims=True) * 100
    for i, category in enumerate(categories):
        model1_plot = plt.bar(x - (width+0.1)/2, percentages_normalized[:, i], bottom=bottom, label=f'{model1_label} - {category}', 
                              linestyle=model1_style, color=category_colors[i], zorder=1, width=width, edgecolor=edgecolor)
        bottom += percentages_normalized[:, i]


    # Plot the first stacked columns
    bottom = np.zeros(len(x_ticks))
    percentages_normalized = model2_data / np.array(model2_data).sum(axis=1, keepdims=True) * 100
    for i, category in enumerate(categories):
        model2_plot = plt.bar(x + (width+0.1)/2, percentages_normalized[:, i], bottom=bottom, label=f'{model2_label} - {category}',  
                              linestyle=model2_style, color=category_colors[i], zorder=1, width=width, edgecolor=edgecolor)
        bottom += percentages_normalized[:, i]
  
    plt.grid(axis='y', color='0.8', linestyle='--')
    # set the x label values
    plt.xticks(x, x_ticks)
    
    # # set the limit for x axis
    # plt.xlim(-2, len(x_ticks))
    
    # set the limit for y axis
    # ax.set_ylim(y_min_value, y_max_value)

    # Set axes and legend
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend(loc='upper left')

    plt.savefig(save_file, dpi=300)



def draw_stacked_column_chart(categories, models, percentages, save_file, x_axis_label=None, y_axis_label=None, title=None,
                    pdf_x_dim = 6.0, pdf_y_dim = 4.0, category_colors = ['skyblue', 'lightcoral'], FS = 10, y_ticks=None):
    '''
    categories: eg. ['Cat1', 'Cat2']
    models: or the x-axis values eg. ['Model 1', 'Model 2', 'Model 3', 'Model 4']
    percentages: the percentage values of each category. Eg. np.array([[30, 70], [40, 60], [20, 80], [25, 75]])

    '''
    # Set font size
    params = {'axes.labelsize': FS,'axes.titlesize': FS, 'legend.fontsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS}
    plt.rcParams.update(params)
    # zorder to choose the order of drawing, i.e.,  0 means draw the grid first behind the bars
    
    # Subplots can be used to put multiple plots in one frame
    fig, ax = plt.subplots(figsize=(pdf_x_dim, pdf_y_dim))

    # Remove the top and right borders on the box around the figure
    ax.spines[['right', 'top']].set_visible(False)

    # Also make sure there are no ticks on the top or right
    ax.tick_params(which="both", top=False, right=False)

    # Add a grid on top of the ticks for improved readibility
    plt.grid(axis='y', color='0.8', linestyle='--', zorder=0)
    # Normalize percentages to ensure they add up to 100 for each model
    percentages_normalized = percentages / np.array(percentages).sum(axis=1, keepdims=True) * 100

    # Plot stacked columns
    bottom = np.zeros(len(models))
    for i, category in enumerate(categories):
        plt.bar(models, percentages_normalized[:, i], bottom=bottom, label=category, color=category_colors[i], zorder=1)
        bottom += percentages_normalized[:, i]

    # Set axes and legend
    plt.yticks(y_ticks)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend(loc='upper left')



    # Output as a PDF
    plt.savefig(save_file, dpi=300)

