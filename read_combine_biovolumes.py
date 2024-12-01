from pathlib import Path
import pandas as pd
from tqdm import tqdm  # Progress bar library

# Directory containing the feature files
feature_dir = Path('/cfs/klemming/projects/supr/snic2020-6-126/projects/amime/manually_classified_ifcb_sets/tangesund/IFCB_Tangesund/features/2016')

# Collect all CSV files ending with '_fea_v2.csv'
filenames = list(feature_dir.glob('*_fea_v2.csv'))

# Initialize an empty DataFrame to store combined results
combined_df = pd.DataFrame()

# Loop through files with a progress bar
for file in tqdm(filenames, desc="Processing files"):
    try:
        # Read each file
        df = pd.read_csv(file)
        # Add the sample name as a column
        df['Sample Name'] = file.stem
        # Append to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Save the combined DataFrame to a CSV file
output_file = feature_dir / 'tangesund_features.csv'
combined_df.to_csv(output_file, index=False)

print(f"Combined file saved as {output_file}")
