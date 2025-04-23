import numpy as np
import pandas as pd

# Data dictionary with "FAIL" replaced as None (or NaN)
data = {
    "Rigid_T20": {
        "Point-to-Surface Mean Distance": [0.009764677460367402, None, 0.009884186941082515, 0.010968030437865853, 0.009922322327397875],
        "Point-to-Surface Max Distance": [0.04060388450990469, None, 0.045744239725584744, 0.05786297669163959, 0.05999445356917391],
        "Overlap Ratio": [0.9813874788494078, None, 0.9839255499153976, 0.9771573604060914, 0.9796954314720813]
    },
    "Rigid_R45": {
        "Point-to-Surface Mean Distance": [0.012639843240520034, None, None, 0.012775654003565565, 0.01328235192707837],
        "Point-to-Surface Max Distance": [0.04945927089835689, None, None, 0.051505059144710784, 0.05712534248630413],
        "Overlap Ratio": [0.9094754653130288, None, None, 0.9077834179357022, 0.9027072758037225]
    },
    "Tilt_30": {
        "Point-to-Surface Mean Distance": [None, 0.01482138030601108, 0.01435331022732006, None, 0.015054323675328237],
        "Point-to-Surface Max Distance": [None, 0.058938390146951354, 0.11714950082225424, None, 0.11591139810105364],
        "Overlap Ratio": [None, 0.9746192893401016, 0.9780033840947546, None, 0.9763113367174281]
    },
    "Generalization": {
        "Point-to-Surface Mean Distance": [0.011338122822821012, 0.011199049093002504, 0.0111174633617318, 0.01118854480487476, 0.01127655458529622],
        "Point-to-Surface Max Distance": [0.05150577529306278, 0.08032813945623783, 0.07186380619164962, 0.07456708735688328, 0.06648876376121643],
        "Overlap Ratio": [1.0, 1.0, 1.0, 1.0, 1.0]
    }
}

# Function to compute mean and standard deviation while ignoring None values
def compute_stats(data_dict):
    results = []
    for scenario, metrics in data_dict.items():
        for metric, values in metrics.items():
            filtered_values = [v for v in values if v is not None]  # Exclude failed cases
            mean_val = np.mean(filtered_values)
            std_val = np.std(filtered_values)
            results.append([scenario, metric, mean_val, std_val])
    return results

# Compute statistics
stats_results = compute_stats(data)

# Convert results into a DataFrame for readability
df_results = pd.DataFrame(stats_results, columns=["Scenario", "Metric", "Mean", "Std Dev"])

# Save results as CSV
csv_filename = "point_cloud_registration_accuracy.csv"
df_results.to_csv(csv_filename, index=False)

# Print results for verification
print(df_results)

print(f"\nResults saved to {csv_filename}")
