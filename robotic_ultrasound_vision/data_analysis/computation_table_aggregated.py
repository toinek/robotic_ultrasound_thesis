import os
import pandas as pd

# Define the directory containing the CSV files
csv_directory = "/home/toine/catkin_ws/src/realsense/data_analysis/computation_csvs"  # Adjust to the actual path

# Define scenario categories based on filename patterns
scenario_categories = {
    "Generalization": lambda filename: filename.startswith("14feb"),
    "Non-Rigid": lambda filename: filename.startswith("comp"), # or "18feb" in filename.lower(),
    "Rigid": lambda filename: filename.startswith("T20"),
}

# Define the categories for computation components
categories = {
    "Image Processing": [
        "RS/2D Object Detection",
        "RS/Bounding Box-based Point Cloud Filtering",
        "RS/Frame Extraction",
        "RS/Point Cloud Transformation/Alignment",
        "RS_mediapipe/Frame Extraction",
        "RS_mediapipe/Keypoint Extraction",
        "RS_mediapipe/PCD Publishing",
        "RS_mediapipe/Point Cloud Generation",
        "RS_mediapipe/Point Cloud Transformation/Alignment"
    ],
    "Point Cloud Registration": [
        "PCD_rigid/Coarse Alignment",
        "PCD_rigid/Downsampling & Chamfer Distance",
        "PCD_rigid/ICP Fine Alignment",
        "PCD_rigid/PCD Message Handling & Preprocessing",
        "PCD_rigid/Reordering Point Clouds",
        "PCD_dynamic/CPD Registration",
        "PCD_dynamic/Downsampling & Chamfer Distance",
        "PCD_dynamic/ICP Fine Alignment",
        "PCD_dynamic/PCD Message Handling & Preprocessing",
        "PCD_dynamic/Reordering Point Clouds",
        "PCD_keypoints/Keypoints entire pipeline"
    ],
    "LTE-Based Trajectory Adaptation": [
        "LTE/Fitting",
        "LTE/Trajectory Update"
    ]
}

# Initialize a dictionary to store structured results
structured_summary = {
    "Component": [],
    "Rigid (ms)": [], "Generalization (ms)": [], "Non-Rigid (ms)": [],
    "Mean Count Rigid": [], "Mean Count Generalization": [], "Mean Count Non-Rigid": []
}

# Iterate over all CSV files in the directory
scenario_data = {scenario: {} for scenario in scenario_categories.keys()}

for filename in os.listdir(csv_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(file_path)

        # Identify the scenario category for this file
        scenario = None
        for category, condition in scenario_categories.items():
            if condition(filename):
                scenario = category
                break

        # Skip file if it does not match any scenario
        if scenario is None:
            continue

        # Aggregate mean times and counts for each processing category
        for category_name, components in categories.items():
            subset = df[df["Component"].isin(components)]
            mean_time = subset["mean"].sum() * 1000  # Convert seconds to milliseconds
            mean_count = subset["count"].mean()  # Compute mean count across trials

            if category_name not in scenario_data[scenario]:
                scenario_data[scenario][category_name] = {"Mean Time": mean_time, "Mean Count": mean_count}
            else:
                scenario_data[scenario][category_name]["Mean Time"] += mean_time
                scenario_data[scenario][category_name]["Mean Count"] += mean_count

# Populate the structured summary
for component in categories.keys():
    structured_summary["Component"].append(component)

    for scenario in ["Rigid", "Generalization", "Non-Rigid"]:
        if component in scenario_data[scenario]:
            structured_summary[f"{scenario} (ms)"].append(scenario_data[scenario][component]["Mean Time"])
            structured_summary[f"Mean Count {scenario}"].append(scenario_data[scenario][component]["Mean Count"])
        else:
            structured_summary[f"{scenario} (ms)"].append(None)
            structured_summary[f"Mean Count {scenario}"].append(None)

# Convert summary data into a DataFrame
structured_summary_df = pd.DataFrame(structured_summary)

# Save the structured CSV file
structured_summary_file = "computation_results/structured_scenario_computation.csv"
structured_summary_df.to_csv(structured_summary_file, index=False)

print(f"Structured scenario computation summary saved to {structured_summary_file}")
