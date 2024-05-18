import shutil
import os
import matplotlib.pyplot as plt
import farver
import viz0
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import norm

rod = 'reports/figures/Eksperimenter/2/fordeling_af_MAE'

if os.path.exists(rod):
    shutil.rmtree(rod)



groups, runs = viz0.get_groups_runs('eksp4')
for group in tqdm(groups):
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    assert len(fortræningsudgaver) == 1
    fortræningsudgave = list(fortræningsudgaver)[0]
    kørsel_path = os.path.join(rod, group)
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

            bins = 5
            plt.hist(df['test_loss_mean'], bins=bins, density=True, alpha=0.4, color=farver.rød, edgecolor='black')  # Histogram
            plt.scatter(x=df['test_loss_mean'], y=[0.0005]*len(df), color=farver.blå, marker='x', s=200)

            plt.plot(x, fit, farver.rød, label='Normalfordeling', linewidth=5)  # Fittet normalfordeling
            plt.xlabel("MAE", fontsize=18)
            plt.ylabel('Frekvens', fontsize=18)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tick_params(axis='both', which='minor', labelsize=14)
            plt.title('Fordeling af MAE', fontsize=22)
            # plt.legend(fontsize=18)
            plt.grid(True)
            plt.savefig(os.path.join(kørsel_path, "fordeling_af_MAE.jpg"))
            plt.savefig(os.path.join(kørsel_path, "fordeling_af_MAE.pdf"))
            plt.close()


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
