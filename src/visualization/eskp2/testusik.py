import random
import shutil
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from src.visualization.farver import farver
from src.visualization import viz0
import pandas as pd
from tqdm import tqdm
import wandb

ROOT = 'reports/figures/Eksperimenter/2/testusik'

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9 fortræning',
            '3D-EMGP-lokalt': '3D-EMGP kun lokalt',
            '3D-EMGP-globalt': '3D-EMGP kun globalt',
            '3D-EMGP-begge': '3D-EMGP'
            }

udvalte = ['eksp2_88', 'eksp2_83']
runs = wandb.Api().runs("afhandling")
runs = list(filter(lambda w: viz0.is_suitable(w, 'eksp2'), runs))

# groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(udvalte):
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    print(temperaturer)
    print(seeds)
    seeds = random.sample(list(seeds), k=3)
    print(seeds)
    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)
    for temperatur in temperaturer:
        ncols = 3
        nrows = len(seeds) // ncols + (1 if len(seeds) % ncols != 0 else 0)
        fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 6 * nrows))
        axs = axs.ravel()

        for idx, seed in enumerate(seeds):
            ax = axs[idx]
            row = idx // ncols
            col = idx % ncols
            # ax = axs[row, col]

            i = 0
            for fortræningsudgave in fortræningsudgaver:
                runs_filtered = list(
                    filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave, seed), runs_in_group))
                df = viz0.get_df(runs_filtered)
                # df = df.apply(pd.to_numeric, errors='coerce')
                # df = df.dropna(how='any')
                label = LABELLER[fortræningsudgave]
                ax.scatter(df["eftertræningsmængde"], df[f"test_loss_mean"], label=label, color=farver[i])
                ax.fill_between(df["eftertræningsmængde"], df[f"test_loss_lower"], df[f"test_loss_upper"],
                                color=farver[i], alpha=0.3)
                i += 1
            ax.set_xlabel("Datamængde", fontsize=25)
            ax.set_ylabel("MAE", fontsize=25)
            ax.set_yscale("log")
            ax.legend(fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.tick_params(axis='both', which='minor', labelsize=13)
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            # ax.grid(True)

        # Fjern tomme subplots, hvis der er nogen
        for j in range(idx + 1, nrows * ncols):
            fig.delaxes(axs.flat[j])
        try:
            plt.tight_layout()
        except ValueError:
            a = 2
        for ext in ['jpg', 'pdf']:
            plt.savefig(os.path.join(kørsel_path, f"{temperatur}_testusik.{ext}"))
        plt.close()




        # ncols = 3
        # nrows = len(seeds) // ncols + 1
        # for seed in seeds:
        #         plt.figure(figsize=(10, 6))
        #         i = 0
        #         for fortræningsudgave in fortræningsudgaver:
        #             runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave, seed), runs_in_group))
        #             df = viz0.get_df(runs_filtered)
        #             df = df.apply(pd.to_numeric, errors='coerce')
        #             df = df.dropna(how='any')
        #             prefix = f'{fortræningsudgave}'
        #             plt.scatter(df["eftertræningsmængde"], df[f"test_loss_mean"], label=prefix, color=farver[i])
        #             plt.fill_between(df["eftertræningsmængde"], df[f"test_loss_lower"], df[f"test_loss_upper"],
        #                              color=farver[i],
        #                              alpha=0.3)
        #             i += 1
        #         plt.title(f'{group} {temperatur} {seed}')
        #         plt.xlabel("Datamængde")
        #         plt.ylabel("MAE")
        #         plt.yscale("log")
        #         plt.legend()
        #         plt.grid(True)
        #         plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{seed}.jpg"))
        #         plt.close()
