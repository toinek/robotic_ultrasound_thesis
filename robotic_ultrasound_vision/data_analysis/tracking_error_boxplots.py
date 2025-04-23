import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define all metrics
metrics = {
    "Mean Tracking Error (m)": {
        "Static": [0.0177, 0.0175, 0.0178, 0.0176, 0.0176],
        "Translate 20": [0.0251, 0.0305, 0.0466, 0.0419, 0.0381],
        "Rotate 45": [0.0189, np.nan, np.nan, 0.0266, 0.0247],
        "Tilt 30": [np.nan, 0.0249, 0.0235, np.nan, 0.0231],
        "Generalization": [0.0465, 0.0454, 0.0432, 0.04, 0.0365],
        "Non-Rigid": [0.0193, 0.0317, 0.0564, 0.0611, 0.0464]
    },
    "Max Tracking Error (m)": {
        "Static": [0.0793, 0.0796, 0.0792, 0.0791, 0.0791],
        "Translate 20": [0.2237, 0.0932, 0.1897, 0.1312, 0.159],
        "Rotate 45": [0.0616, np.nan, np.nan, 0.095, 0.0774],
        "Tilt 30": [np.nan, 0.068, 0.0865, np.nan, 0.0731],
        "Generalization": [0.1772, 0.1916, 0.1936, 0.1773, 0.1578],
        "Non-Rigid": [0.0558, 0.2266, 0.4859, 0.6818, 0.3382]
    },
    "Mean Jerk (m/s³)": {
        "Static": [3.873526, 4.073154, 3.96614, 4.071075, 4.145764],
        "Translate 20": [4.010435, 4.031457, 16.733826, 3.955087, 22.385838],
        "Rotate 45": [18.413712, 4.948404, 17.708247, 3.805989, 3.536276],
        "Tilt 30": [3.875416, 3.886442, np.nan, 3.643471, 6.61733175],
        "Generalization": [4.392734, 5.089393, 4.058201, 4.675073, 4.674251],
        "Non-Rigid": [np.nan, 4.991297, 17.937977, 20.799464, 17.177548]
    },
    "Max Jerk (m/s³)": {
        "Static": [17.886632, 21.266918, 19.172039, 20.687838, 19.73302],
        "Translate 20": [20.605993, 19.6273, 545.687408, 20.190768, 578.617798],
        "Rotate 45": [525.636659, 77.967547, 441.05113, 19.040128, 18.616086],
        "Tilt 30": [18.325504, 17.756399, np.nan, 21.441667, 123.1456148],
        "Generalization": [28.622737, 33.069702, 20.561275, 26.309349, 27.64138],
        "Non-Rigid": [np.nan, 29.415662, 291.229496, 646.610566, 363.378461]
    }
}

# Function to generate a box plot for a selected metric
def plot_metric(metric_name):
    if metric_name not in metrics:
        print(f"Metric '{metric_name}' not found. Available metrics: {list(metrics.keys())}")
        return

    df = pd.DataFrame(metrics[metric_name])  # Convert dictionary to DataFrame

    plt.figure(figsize=(60, 36))
    plt.boxplot(
        [df[col].dropna() for col in df.columns],
        labels=df.columns,
        boxprops=dict(linewidth=3),  # Thicker box lines
        medianprops=dict(linewidth=2),  # Thicker median line
        whiskerprops=dict(linewidth=2),  # Thicker whiskers
        capprops=dict(linewidth=2),  # Thicker caps
        flierprops=dict(marker='o', markersize=6, linestyle='none', markeredgewidth=2)  # Outliers
    )

    # Formatting
    # plt.title(f"Box Plot of {metric_name} Across Scenarios", fontsize=16)
    plt.ylabel(metric_name, fontsize=40)
    # plt.xlabel("Scenarios", fontsize=30)
    plt.xticks(fontsize=30, rotation=10)
    plt.yticks(fontsize=30)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show plot
    plt.show()

# Example usage: Choose the metric you want to plot
plot_metric("Max Jerk (m/s³)")
