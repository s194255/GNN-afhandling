import shutil
import os
import matplotlib.pyplot as plt
from farver import farver
import viz0
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import norm

if os.path.exists("lightning_logs"):
    shutil.rmtree("lightning_logs")



groups, runs = viz0.get_groups_runs('eksp4')
for group in tqdm(groups):
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    assert len(fortræningsudgaver) == 1
    fortræningsudgave = list(fortræningsudgaver)[0]
    kørsel_path = os.path.join("lightning_logs", group)
    os.makedirs(kørsel_path)
    for temperatur in temperaturer:
        for seed in seeds:
            runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave, seed), runs_in_group))
            df = viz0.get_df(runs_filtered)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='any')
            prefix = f'{fortræningsudgave}'

            mean = df['test_loss_mean'].mean()
            std_dev = df['test_loss_mean'].std()

            x = np.linspace(df['test_loss_mean'].min(), df['test_loss_mean'].max(), 100)
            fit = norm.pdf(x, mean, std_dev)

            plt.figure(figsize=(10, 6))

            # plt.hist(df['test_loss_mean'], bins=20, density=True, alpha=0.6, color='g', edgecolor='black')  # Histogram
            plt.scatter(x=df['test_loss_mean'], y=[0.0005]*len(df))

            plt.plot(x, fit, '-r', label='Normalfordeling')  # Fittet normalfordeling
            plt.xlabel('Test Loss Mean')
            plt.ylabel('Frekvens')
            plt.title('Fordeling af Test Loss Mean med Fittet Normalfordeling')
            plt.legend()
            plt.grid(True)
            plt.show()


            # plt.figure(figsize=(10, 6))
            # plt.scatter(df["eftertræningsmængde"], df["test_loss_mean"], label=prefix, color=farver[i])
            # plt.fill_between(df["eftertræningsmængde"], df[f"test_loss_lower"], df[f"test_loss_upper"],
            #                  color=farver[0],
            #                  alpha=0.3)
            # plt.title(f'{group} {temperatur} {seed}')
            # plt.xlabel("Datamængde")
            # plt.ylabel("MAE")
            # plt.yscale("log")
            # plt.legend()
            # plt.grid(True)
            # plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{seed}.jpg"))
            # plt.close()
