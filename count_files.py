#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains a method for counting files in subdirectories, and plotting the results. This is used to display the imbalance of the datasets
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

base_dir = Path("/proj/berzelius-2023-48/ifcb/main_folder_karin")

# these paths should be updated :))
# Set the path to your main folder
path = '/proj/berzelius-2023-48/ifcb/main_folder_karin/data/SYKE_2022/labeled_20201020'


# Initialize two empty lists to store the subdirectory names and the number of PNG files in each
dirs = []
counts = []

# Loop through each subdirectory in the main folder
for subdir in os.listdir(path):
    # Check if the current item in the loop is a directory
    if os.path.isdir(os.path.join(path, subdir)):
        # Count the number of PNG files in the current subdirectory
        png_count = len([file for file in os.listdir(os.path.join(path, subdir)) if file.endswith('.png')])
        # Append the subdirectory name and PNG file count to their respective lists
        dirs.append(subdir)
        counts.append(png_count)

# Combine the dirs and counts lists into a dictionary
data = {"Subdirectory": dirs, "PNG Count": counts}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame as a CSV file
df.to_csv(base_dir / 'out' / 'count_table.csv', index=False)

# Sort the subdirectories and counts by the count in descending order
dirs, counts = zip(*sorted(zip(dirs, counts), key=lambda x: x[1], reverse=True))



# Create a bar chart of the counts with the subdirectory names on the x-axis
fig, ax = plt.subplots(figsize=(20, 8))
ax.bar(dirs, counts)
#ax.set_xlabel('Subdirectory', fontsize=9, rotation=90)
ax.set_ylabel('Number of images')

# Set the size and rotation of the x-axis labels
plt.xticks(fontsize=8, rotation=90)
ax.set_yscale('log')

# Save the plot to the specified directory
plt.savefig(base_dir / 'out' / 'class_imbalance.png', bbox_inches='tight')

