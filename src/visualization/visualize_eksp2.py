import pandas as pd
import matplotlib.pyplot as plt
import os

hoved_kørsel_path = "eksp2_logs_hpc"
kørsler = os.listdir(hoved_kørsel_path)

for kørsel in kørsler:
    kørsel_path = os.path.join(hoved_kørsel_path, kørsel)
    df = pd.read_csv(os.path.join(kørsel_path, "logs_metrics.csv"))

    if kørsel in ['kørsel_3', 'kørsel_4', 'kørsel_5']:
        df = df.iloc[1:]

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot uden_test_loss
    plt.plot(df["datamængde"], df["uden_test_loss_mean"], label="uden", color="#2F3EEA")
    plt.fill_between(df["datamængde"], df["uden_test_loss_lower"], df["uden_test_loss_upper"], color="#2F3EEA", alpha=0.3)

    # Plot med_test_loss
    plt.plot(df["datamængde"], df["med_test_loss_mean"], label="med", color="#990000")
    plt.fill_between(df["datamængde"], df["med_test_loss_lower"], df["med_test_loss_upper"], color="#990000", alpha=0.3)

    # Tittel og labels
    plt.title("Testtab")
    plt.xlabel("Datamængde")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(kørsel_path, "results.jpg"))
    if kørsel == 'kørsel_2':
        plt.savefig("../figurer/resultater/eksp2.pdf")
