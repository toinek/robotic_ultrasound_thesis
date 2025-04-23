import csv
import os
import time
import threading
import pandas as pd

class TimingLogger:
    _instance = None
    _lock = threading.Lock()
    LOG_FILE = "/home/toine/catkin_ws/computation_times.csv"

    def __new__(cls):
        """Ensure only one instance of the logger exists"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check locking
                    cls._instance = super().__new__(cls)
                    cls._instance.init_logger()
        return cls._instance

    def init_logger(self):
        """Initialize the CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Component", "Duration (s)"])  # Column headers

    def log_timing(self, component, duration):
        """Log computation time to a CSV file"""
        with self._lock:  # Ensure thread safety
            with open(self.LOG_FILE, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), component, duration])

    def compute_average_per_timestamp(self):
        """Reads the log file and computes the average computation time per component for each timestamp."""
        if not os.path.exists(self.LOG_FILE):
            print("Log file not found.")
            return

        # Load the CSV
        df = pd.read_csv(self.LOG_FILE)

        # Convert "Timestamp" to datetime format
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Compute the average duration per timestamp for each component
        df_avg = df.groupby(["Timestamp", "Component"])["Duration (s)"].mean().unstack()

        # Fill missing values with 0 for clarity
        df_avg = df_avg.fillna(0)

        # Print table
        print("\nComputation Time Averages Per Timestamp:")
        print(df_avg)

        # Save as CSV for further analysis
        output_file = "/home/toine/catkin_ws/computation_times_avg.csv"
        df_avg.to_csv(output_file)
        print(f"\nSaved computation averages to: {output_file}")

    def compute_metrics(self):
        """Reads the log file and computes the mean, standard deviation, and occurrence count for each component."""
        if not os.path.exists(self.LOG_FILE):
            print("Log file not found.")
            return

        # Load the CSV
        df = pd.read_csv(self.LOG_FILE)

        # Convert "Timestamp" to datetime format
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Compute mean, standard deviation, and occurrence count for each component
        metrics = df.groupby("Component")["Duration (s)"].agg(["mean", "std", "count"])

        # Save metrics to a CSV file
        output_file = "/home/toine/catkin_ws/14feb_generalize_comp_7.csv"
        metrics.to_csv(output_file)

        # Print the table
        print("\nComputation Time Metrics (Mean ± Std, Count):")
        print(metrics)

        print(f"\nSaved computation time metrics to: {output_file}")

# Singleton instance
timing_logger = TimingLogger()

# Run analysis if the script is executed directly
if __name__ == "__main__":
    timing_logger.compute_metrics()