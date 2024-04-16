import pandas as pd
import matplotlib.pyplot as plt
import os
from farver import farvekort, farver

hoved_kørsel_path = "eksp2_logs_hpc/logging"
kørsler = os.listdir(hoved_kørsel_path)

for kørsel in kørsler:
    kørsel_path = os.path.join(hoved_kørsel_path, kørsel)
    df = pd.read_csv(os.path.join(kørsel_path, "logs_metrics.csv"))



    # Plot
    for frys_rygrad in [True, False]:
        plt.figure(figsize=(10, 6))
        i = 0
        for mode in ['med', 'uden']:
            prefix = f'{mode}_{frys_rygrad}'
            plt.plot(df["datamængde"], df[f"{prefix}_test_loss_mean"], label=prefix, color=farver[i])
            plt.fill_between(df["datamængde"], df[f"{prefix}_test_loss_lower"], df[f"{prefix}_test_loss_upper"], color=farver[i],
                             alpha=0.3)
            i += 1


        # Tittel og labels
        plt.title(f'{kørsel} {frys_rygrad}')

        plt.xlabel("Datamængde")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(kørsel_path, f'{kørsel}_{frys_rygrad}.jpg'))
        # if kørsel == 'kørsel_2':
        #     plt.savefig(os.path.join("..", "figurer", "resultater", f"{kørsel}.pdf"))