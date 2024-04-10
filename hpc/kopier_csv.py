import os

rod = "eksk2_logs"

for kørsel in os.listdir():
    csv_path_hpc = os.path.join(rod, kørsel, "logs_metrics.csv")
    csv_path_local = ""