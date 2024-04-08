import pandas as pd
import matplotlib.pyplot as plt

# Antag at data er læst fra en CSV-fil og gemt i en DataFrame kaldet df
# df = pd.read_csv('din_fil.csv')

# Eksempeldata
data = {
    "uden_test_loss_mean": [0.5, 0.6, 0.7, 0.8, 0.9],
    "uden_test_loss_lower": [0.4, 0.5, 0.6, 0.7, 0.8],
    "uden_test_loss_upper": [0.6, 0.7, 0.8, 0.9, 1.0],
    "med_test_loss_mean": [0.4, 0.5, 0.6, 0.7, 0.8],
    "med_test_loss_lower": [0.3, 0.4, 0.5, 0.6, 0.7],
    "med_test_loss_upper": [0.5, 0.6, 0.7, 0.8, 0.9],
    "datamængde": [100, 200, 300, 400, 500]
}

# df = pd.DataFrame(data)
df = pd.read_csv("logs_metrics.csv")

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
plt.show()
