import os
import pandas as pd

# Define the directory containing the CSV files
csv_directory = "/home/toine/catkin_ws/src/realsense/data_analysis/computation_csvs"  # Adjust to the actual path

# Define scenario categories based on filename patterns
scenario_categories = {
    "Generalization": lambda filename: filename.startswith("14feb"),
    "Non-Rigid": lambda filename: filename.startswith("comp"),  # or "18feb" in filename.lower(),
    "Rigid": lambda filename: filename.startswith("T20"),
}

# List of all individual computation components
components = [
    "RS/2D Object Detection",
    "RS/Bounding Box-based Point Cloud Filtering",
    "RS/Frame Extraction",
    "RS/Point Cloud Transformation/Alignment",
    "RS_mediapipe/Frame Extraction",
    "RS_mediapipe/Keypoint Extraction",
    "RS_mediapipe/PCD Publishing",
    "RS_mediapipe/Point Cloud Generation",
    "RS_mediapipe/Point Cloud Transformation/Alignment",
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
    "PCD_keypoints/Keypoints entire pipeline",
    "LTE/Fitting",
    "LTE/Trajectory Update"
]

# Initialize a dictionary to store computation time and count for each scenario
scenario_data = {
    scenario: {comp: {"total_time": 0, "total_count": 0} for comp in components}
    for scenario in scenario_categories.keys()
}

# Iterate over all CSV files in the directory
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

        # Process each individual component separately
        for _, row in df.iterrows():
            component = row["Component"]
            mean_time = row["mean"] * row["count"] * 1000  # Convert seconds to milliseconds
            count = row["count"]

            if component in scenario_data[scenario]:
                scenario_data[scenario][component]["total_time"] += mean_time
                scenario_data[scenario][component]["total_count"] += count

# Compute final weighted averages
structured_summary = {
    "Component": [],
    "Rigid (ms)": [], "Generalization (ms)": [], "Non-Rigid (ms)": []
}

for component in components:
    structured_summary["Component"].append(component)

    for scenario in ["Rigid", "Generalization", "Non-Rigid"]:
        total_time = scenario_data[scenario][component]["total_time"]
        total_count = scenario_data[scenario][component]["total_count"]

        if total_count > 0:
            structured_summary[f"{scenario} (ms)"].append(total_time / total_count)
        else:
            structured_summary[f"{scenario} (ms)"].append(0)

# Convert summary data into a DataFrame
structured_summary_df = pd.DataFrame(structured_summary)

# Save the structured CSV file
structured_summary_file = "computation_results/structured_scenario_computation_detailed.csv"
structured_summary_df.to_csv(structured_summary_file, index=False)

print(f"Detailed structured scenario computation summary saved to {structured_summary_file}")
