import pandas as pd
import matplotlib.pyplot as plt
import re
import random

def plot_tax_classes(relative_abundance_per_class, is_svea_data):
    data = relative_abundance_per_class
    # Get the unique bin names and taxonomic classes
    bins = data["bin_name"].unique()
    taxonomic_classes = data["taxonomic_class"].unique()

    # Create a colormap with a unique color for each class
    num_classes = len(taxonomic_classes)
    cmap = plt.cm.get_cmap('Paired', num_classes)

    # Map each class name to a color in the colormap
    class_colors = {class_name: cmap(i/num_classes) for i, class_name in enumerate(taxonomic_classes)}

    # Randomize the mapping of classes to colors
    random.seed(123)
    class_names = list(class_colors.keys())
    random.shuffle(class_names)
    class_colors = {class_name: class_colors[class_names[i]] for i, class_name in enumerate(taxonomic_classes)}

    # Extract the dates from the bin names using regular expressions
    dates = [re.findall(r'D(\d{4})(\d{2})(\d{2})T', bin_name)[0] for bin_name in bins]
    dates = [f'{y}-{m}-{d}' for y, m, d in dates]

    # Group the bins by date
    bins_by_date = {}
    for i, date in enumerate(dates):
        if date not in bins_by_date:
            bins_by_date[date] = []
        bins_by_date[date].append(i)


    ## PLOT 1: relative abundance with unclassifiable

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a pivot table to reshape the data
    pivot_data_rel_abundance = pd.pivot_table(data, values='relative_abundance', index=['bin_name'], columns=['taxonomic_class'], fill_value=0)

    # Filter out taxonomic classes that have a maximum relative abundance value of 0.01 or lower
    filtered_classes = []
    for tc in taxonomic_classes:
        if pivot_data_rel_abundance[tc].max() > 0.01:
            filtered_classes.append(tc)

    # Plot each class as a separate line
    for tc in filtered_classes:
        ax.plot(bins, pivot_data_rel_abundance[tc], color=class_colors[tc], label=tc)

    # Set the x-axis label and tick labels
    # ax.set_xlabel("Bin Name")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([], rotation=90, fontsize=5)

    # Annotate the plot with arrowed ranges for each date
    for date, indices in bins_by_date.items():
        start = indices[0]
        end = indices[-1]
        ax.annotate("", xy=(end+1+1, 0), xytext=(start, 0), arrowprops=dict(arrowstyle='|-|'))
        ax.annotate(date, xy=(start + (end-start)/2, -0.05), ha='center', va='top', fontsize=8, annotation_clip=False)

    if is_svea_data:
        # Add arrow and text for stop due to rough seas
        bin_name = 'D20230312T040242_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("11 h stop due\nto rough seas", xy=(index, pivot_data_rel_abundance.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_rel_abundance.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T130251_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nÖresund", xy=(index, pivot_data_rel_abundance.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_rel_abundance.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')


        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T213259_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nKattegat", xy=(index, pivot_data_rel_abundance.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_rel_abundance.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')

    # Set the y-axis label
    ax.set_ylabel("Relative Abundance")
    ax.set_ylim(0, 1)

    # Set the title and legend
    ax.set_title("Relative Abundance by Taxonomic Class (with unclassified images)")
    ax.legend()

    plt.tight_layout()

    # Save the plot as an image
    plt.savefig("out/plots_on_svea_data/relative_abundance_by_tax_class_with_uncl.png", dpi=300)

    # PLOT 2: counts with unclassifiable

    # Create a pivot table to reshape the data
    pivot_data_counts = pd.pivot_table(data, values='counts_per_bin', index=['bin_name'], columns=['taxonomic_class'], fill_value=0)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each class as a separate line
    for tc in filtered_classes:
        ax.plot(bins, pivot_data_counts[tc], color=class_colors[tc], label=tc)

    # Set the x-axis label and tick labels
    # ax.set_xlabel("Bin Name")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([], rotation=90, fontsize=5)

    # Annotate the plot with arrowed ranges for each date
    for date, indices in bins_by_date.items():
        start = indices[0]
        end = indices[-1]
        ax.annotate("", xy=(end+1+1, -120), xytext=(start, -120), arrowprops=dict(arrowstyle='|-|'))
        ax.annotate(date, xy=(start + (end-start)/2, -200), ha='center', va='top', fontsize=8, annotation_clip=False)

    if is_svea_data:
        # Add arrow and text for stop due to rough seas
        bin_name = 'D20230312T040242_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("11 h stop due\nto rough seas", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T130251_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nÖresund", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T213259_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nKattegat", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

    # Set the y-axis label
    ax.set_ylabel("Counts per 5 ml sample")

    # Set the title and legend
    ax.set_title("Counts by Taxonomic Class (with unclassified images)")
    ax.legend()

    plt.tight_layout()

    # Save the plot as an image
    plt.savefig("out/plots_on_svea_data/counts_by_tax_class_with_uncl.png", dpi=300)

    ## PLOT 3: relative abundance without unclassifiable

    # Create a pivot table to reshape the data
    pivot_data_rel_abundance = pd.pivot_table(data, values='relative_abundance_without_unclassifiable', index=['bin_name'], columns=['taxonomic_class'], fill_value=0)

    # Filter out taxonomic classes that have a maximum relative abundance value of 0.01 or lower
    filtered_classes = []
    for tc in taxonomic_classes:
        if pivot_data_rel_abundance[tc].max() > 0.01:
            filtered_classes.append(tc)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each class as a separate line
    for tc in filtered_classes:
        ax.plot(bins, pivot_data_rel_abundance[tc], color=class_colors[tc], label=tc)

    # Set the x-axis label and tick labels
    # ax.set_xlabel("Bin Name")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([], rotation=90, fontsize=5)

    # Annotate the plot with arrowed ranges for each date
    for date, indices in bins_by_date.items():
        start = indices[0]
        end = indices[-1]
        ax.annotate("", xy=(end+1+1, 0), xytext=(start, 0), arrowprops=dict(arrowstyle='|-|'))
        ax.annotate(date, xy=(start + (end-start)/2, -0.05), ha='center', va='top', fontsize=8, annotation_clip=False)

    if is_svea_data:
        # Add arrow and text for stop due to rough seas
        bin_name = 'D20230312T040242_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("11 h stop due\nto rough seas", xy=(index, pivot_data_rel_abundance.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_rel_abundance.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T130251_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nÖresund", xy=(index, pivot_data_rel_abundance.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_rel_abundance.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')


        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T213259_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nKattegat", xy=(index, pivot_data_rel_abundance.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_rel_abundance.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')

    # Set the y-axis label
    ax.set_ylabel("Relative Abundance")
    ax.set_ylim(0, 1)

    # Set the title and legend
    ax.set_title("Relative Abundance by Taxonomic Class (without unclassified images)")
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    lgd = ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))


    plt.tight_layout()

    # Save the plot as an image
    plt.savefig("out/plots_on_svea_data/relative_abundance_by_tax_class_without_uncl.png", dpi=300)
    
    # PLOT 4: counts without unclassifiable

    # Remove the Unclassifiable counts
    data_cleaned = data[data['taxonomic_class'] != 'Unclassified']

    # Create a pivot table to reshape the data
    pivot_data_counts = pd.pivot_table(data_cleaned, values='counts_per_bin', index=['bin_name'], columns=['taxonomic_class'], fill_value=0)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each class as a separate line
    for tc in filtered_classes:
        ax.plot(bins, pivot_data_counts[tc], color=class_colors[tc], label=tc)

    # Set the x-axis label and tick labels
    # ax.set_xlabel("Bin Name")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([], rotation=90, fontsize=5)

    # Annotate the plot with arrowed ranges for each date
    for date, indices in bins_by_date.items():
        start = indices[0]
        end = indices[-1]
        ax.annotate("", xy=(end+1+1, -120), xytext=(start, -120), arrowprops=dict(arrowstyle='|-|'))
        ax.annotate(date, xy=(start + (end-start)/2, -200), ha='center', va='top', fontsize=8, annotation_clip=False)

    if is_svea_data:
        # Add arrow and text for stop due to rough seas
        bin_name = 'D20230312T040242_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("11 h stop due\nto rough seas", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T130251_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nÖresund", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T213259_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nKattegat", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

    # Set the y-axis label
    ax.set_ylabel("Counts per 5 ml sample")

    # Set the title and legend
    ax.set_title("Counts by Taxonomic Class (without unclassified images)")
    ax.legend()

    plt.tight_layout()

    # Save the plot as an image
    plt.savefig("out/plots_on_svea_data/counts_by_tax_class_without_uncl.png", dpi=300)












def plot_image_classes(data, is_svea_data):
    # Get the unique bin names and taxonomic classes
    bins = data["bin_name"].unique()
    taxonomic_classes = data["predicted_class"].unique()

    # Extract the dates from the bin names using regular expressions
    dates = [re.findall(r'D(\d{4})(\d{2})(\d{2})T', bin_name)[0] for bin_name in bins]
    dates = [f'{y}-{m}-{d}' for y, m, d in dates]


    # Create a colormap with a unique color for each class
    num_classes = len(taxonomic_classes)
    cmap = plt.cm.get_cmap('Paired', num_classes)

    # Map each class name to a color in the colormap
    class_colors = {class_name: cmap(i/num_classes) for i, class_name in enumerate(taxonomic_classes)}

    # Randomize the mapping of classes to colors
    random.seed(123)
    class_names = list(class_colors.keys())
    random.shuffle(class_names)
    class_colors = {class_name: class_colors[class_names[i]] for i, class_name in enumerate(taxonomic_classes)}


    # Group the bins by date
    bins_by_date = {}
    for i, date in enumerate(dates):
        if date not in bins_by_date:
            bins_by_date[date] = []
        bins_by_date[date].append(i)


    ## PLOT 1: relative abundance with unclassifiable

    # Create a pivot table to reshape the data
    pivot_data = pd.pivot_table(data, values='relative_abundance', index=['bin_name'], columns=['predicted_class'], fill_value=0)

    # Filter out taxonomic classes that have a maximum relative abundance value of 0.01 or lower
    filtered_classes = []
    for tc in taxonomic_classes:
        if pivot_data[tc].max() > 0.01:
            filtered_classes.append(tc)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each class as a separate line
    for tc in filtered_classes:
        ax.plot(bins, pivot_data[tc], label=tc, color=class_colors[tc])

    # Set the x-axis label and tick labels
    # ax.set_xlabel("Bin Name")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([], rotation=90, fontsize=5)

    # Annotate the plot with arrowed ranges for each date
    for date, indices in bins_by_date.items():
        start = indices[0]
        end = indices[-1]
        ax.annotate("", xy=(end+1+1.5, 0), xytext=(start, 0), arrowprops=dict(arrowstyle='|-|'))
        ax.annotate(date, xy=(start + (end-start)/2, -0.05), ha='center', va='top', fontsize=8, annotation_clip=False)

    if is_svea_data:
        # Add arrow and text for stop due to rough seas
        bin_name = 'D20230312T040242_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("11 h stop due\nto rough seas", xy=(index, pivot_data.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T130251_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nÖresund", xy=(index, pivot_data.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')
        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T213259_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nKattegat", xy=(index, pivot_data.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')
        
    # Set the y-axis label
    ax.set_ylabel("Relative Abundance")
    ax.set_ylim(0, 1)
    # Set the title and legend
    ax.set_title("Relative Abundance by Image Class (with unclassified)")

    # Add a legend
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    lgd = ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    # Save the plot as an image
    fig.savefig('out/plots_on_svea_data/relative_abundance_by_image_class.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # PLOT 2: counts with unclassifiable

    # Create a pivot table to reshape the data
    pivot_data_counts = pd.pivot_table(data, values='counts_per_bin', index=['bin_name'], columns=['predicted_class'], fill_value=0)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each class as a separate line
    for tc in filtered_classes:
        ax.plot(bins, pivot_data_counts[tc], color=class_colors[tc], label=tc)

    # Set the x-axis label and tick labels
    # ax.set_xlabel("Bin Name")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([], rotation=90, fontsize=5)

    # Annotate the plot with arrowed ranges for each date
    for date, indices in bins_by_date.items():
        start = indices[0]
        end = indices[-1]
        ax.annotate("", xy=(end+1+1, -120), xytext=(start, -120), arrowprops=dict(arrowstyle='|-|'))
        ax.annotate(date, xy=(start + (end-start)/2, -200), ha='center', va='top', fontsize=8, annotation_clip=False)

    if is_svea_data:
        # Add arrow and text for stop due to rough seas
        bin_name = 'D20230312T040242_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("11 h stop due\nto rough seas", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 800), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T130251_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nÖresund", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 300), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T213259_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nKattegat", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

    # Set the y-axis label
    ax.set_ylabel("Counts per 5 ml sample")

    # Set the title and legend
    ax.set_title("Counts by Image Class (with unclassified images)")

    # Add a legend
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    lgd = ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    plt.tight_layout()

    # Save the plot as an image
    plt.savefig("out/plots_on_svea_data/counts_by_image_class_with_uncl.png", dpi=300)


    ## PLOT 3: relative abundance without unclassifiable

    # Create a pivot table to reshape the data
    pivot_data_rel_abundance = pd.pivot_table(data, values='relative_abundance_without_unclassifiable', index=['bin_name'], columns=['predicted_class'], fill_value=0)

    # Filter out taxonomic classes that have a maximum relative abundance value of 0.01 or lower
    filtered_classes = []
    for tc in taxonomic_classes:
        if pivot_data_rel_abundance[tc].max() > 0.01:
            filtered_classes.append(tc)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each class as a separate line
    for tc in filtered_classes:
        ax.plot(bins, pivot_data_rel_abundance[tc], color=class_colors[tc], label=tc)

    # Set the x-axis label and tick labels
    # ax.set_xlabel("Bin Name")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([], rotation=90, fontsize=5)

    # Annotate the plot with arrowed ranges for each date
    for date, indices in bins_by_date.items():
        start = indices[0]
        end = indices[-1]
        ax.annotate("", xy=(end+1+1, 0), xytext=(start, 0), arrowprops=dict(arrowstyle='|-|'))
        ax.annotate(date, xy=(start + (end-start)/2, -0.05), ha='center', va='top', fontsize=8, annotation_clip=False)

    if is_svea_data:
        # Add arrow and text for stop due to rough seas
        bin_name = 'D20230312T040242_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("11 h stop due\nto rough seas", xy=(index, pivot_data_rel_abundance.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_rel_abundance.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T130251_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nÖresund", xy=(index, pivot_data_rel_abundance.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_rel_abundance.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')


        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T213259_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nKattegat", xy=(index, pivot_data_rel_abundance.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_rel_abundance.loc[bin_name].max() + 0.3), arrowprops=dict(arrowstyle='->'), ha='center')

    # Set the y-axis label
    ax.set_ylabel("Relative Abundance")
    ax.set_ylim(0, 1)

    # Set the title and legend
    ax.set_title("Relative Abundance by Image Class (without unclassified images)")

    # Add a legend
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    lgd = ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    plt.show()

    # Save the plot as an image
    plt.savefig("out/plots_on_svea_data/relative_abundance_by_image_class_without_uncl.png", dpi=300)


    # PLOT 4: counts without unclassifiable

    # Remove the Unclassifiable counts
    data_cleaned = data[data['predicted_class'] != 'Unclassified']

    # Create a pivot table to reshape the data
    pivot_data_counts = pd.pivot_table(data_cleaned, values='counts_per_bin', index=['bin_name'], columns=['predicted_class'], fill_value=0)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each class as a separate line
    for tc in filtered_classes:
        ax.plot(bins, pivot_data_counts[tc], color=class_colors[tc], label=tc)

    # Set the x-axis label and tick labels
    # ax.set_xlabel("Bin Name")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([], rotation=90, fontsize=5)

    # Annotate the plot with arrowed ranges for each date
    for date, indices in bins_by_date.items():
        start = indices[0]
        end = indices[-1]
        ax.annotate("", xy=(end+1+1, -120), xytext=(start, -120), arrowprops=dict(arrowstyle='|-|'))
        ax.annotate(date, xy=(start + (end-start)/2, -200), ha='center', va='top', fontsize=8, annotation_clip=False)

    if is_svea_data:
        # Add arrow and text for stop due to rough seas
        bin_name = 'D20230312T040242_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("11 h stop due\nto rough seas", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T130251_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nÖresund", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T213259_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nKattegat", xy=(index, pivot_data_counts.loc[bin_name].max()+0.1),
                    xytext=(index, pivot_data_counts.loc[bin_name].max() + 500), arrowprops=dict(arrowstyle='->'), ha='center')

    # Set the y-axis label
    ax.set_ylabel("Counts per 5 ml sample")

    # Set the title and legend
    ax.set_title("Counts by Image Class (without unclassified images)")
    # Add a legend
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    lgd = ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

    #plt.tight_layout()

    # Save the plot as an image
    plt.savefig("out/plots_on_svea_data/counts_by_image_class_without_uncl.png", dpi=300)

    # PLOT 5: IFCB stats
    ifcb_info = pd.read_csv('/proj/berzelius-2023-48/ifcb/main_folder_karin/supportive_files/allifcb_data_wide_march_2023.csv')

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each class as a separate line
    ax.plot(bins, ifcb_info['roiCount'])

    # Set the x-axis label and tick labels
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([], rotation=90, fontsize=5)

    # Annotate the plot with arrowed ranges for each date
    for date, indices in bins_by_date.items():
        start = indices[0]
        end = indices[-1]
        ax.annotate("", xy=(end+1+1, -120), xytext=(start, -120), arrowprops=dict(arrowstyle='|-|'))
        ax.annotate(date, xy=(start + (end-start)/2, -200), ha='center', va='top', fontsize=8, annotation_clip=False)

    if is_svea_data:
        # Add arrow and text for stop due to rough seas
        bin_name = 'D20230312T040242_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("11 h stop due\nto rough seas", xy=(index, 1500),
                    xytext=(index, 1500 + 500), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T130251_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nÖresund", xy=(index, 2300),
                    xytext=(index, 2300 + 500), arrowprops=dict(arrowstyle='->'), ha='center')

        # Add arrow and text for entry into Öresund
        bin_name = 'D20230313T213259_IFCB134'
        index = bins.tolist().index(bin_name)
        ax.annotate("Entry into\nKattegat", xy=(index, 2000),
                    xytext=(index, 2000 + 500), arrowprops=dict(arrowstyle='->'), ha='center')

    # Set the y-axis label
    ax.set_ylabel("Number of ROI:s")

    # Set the title and legend
    ax.set_title("Number of ROI:s per 5 ml sample")

    plt.show()

    # Save the plot as an image
    plt.savefig("out/plots_on_svea_data/total_roi_counts.png", dpi=300)
