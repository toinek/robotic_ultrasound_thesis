import pandas as pd

# Define the input and output file paths
input_csv = "computation_results/structured_scenario_computation_detailed.csv"
output_csv = "computation_results/summed_scenario_computation.csv"

# Load the CSV file
df = pd.read_csv(input_csv)

# Define component categories
categories = {
    "RS": ["RS/", "RS_mediapipe/"],
    "PCD": ["PCD_rigid/", "PCD_dynamic/", "PCD_keypoints/"],
    "LTE": ["LTE/"]
}

# Initialize a dictionary to store summed values
summed_results = {
    "Component": ["RS", "PCD", "LTE"],
    "Rigid (ms)": [0, 0, 0],
    "Generalization (ms)": [0, 0, 0],
    "Non-Rigid (ms)": [0, 0, 0]
}

# Sum up the values for each category
for index, row in df.iterrows():
    component = row["Component"]
    for cat_index, (category, prefixes) in enumerate(categories.items()):
        if any(component.startswith(prefix) for prefix in prefixes):
            summed_results["Rigid (ms)"][cat_index] += row["Rigid (ms)"]
            summed_results["Generalization (ms)"][cat_index] += row["Generalization (ms)"]
            summed_results["Non-Rigid (ms)"][cat_index] += row["Non-Rigid (ms)"]

# Convert to DataFrame
summed_df = pd.DataFrame(summed_results)

# Save the results
summed_df.to_csv(output_csv, index=False)

print(f"Summed scenario computation summary saved to {output_csv}")
