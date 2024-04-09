import pandas as pd
import matplotlib.pyplot as plt
import os

kørsel_path = "eksp2_logs/hpckørsel_5"

df = pd.read_csv(os.path.join(kørsel_path, "logs_metrics.csv"))

df = df.iloc[1:]

# Plot
plt.figure(figsize=(10, 6))

# Plot uden_test_loss
plt.plot(df["datamængde"], df["uden_test_loss_mean"], label="uden", color="blue")
plt.fill_between(df["datamængde"], df["uden_test_loss_lower"], df["uden_test_loss_upper"], color="blue", alpha=0.3)

# Plot med_test_loss
plt.plot(df["datamængde"], df["med_test_loss_mean"], label="med", color="orange")
plt.fill_between(df["datamængde"], df["med_test_loss_lower"], df["med_test_loss_upper"], color="orange", alpha=0.3)

# Tittel og labels
plt.title("Test Loss")
plt.xlabel("Datamængde")
plt.ylabel("Test Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(kørsel_path, "results.jpg"))
