import random
import shutil
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from src.visualization.farver import farver
from src.visualization import viz0
import pandas as pd
from tqdm import tqdm
import copy
import numpy as np
from itertools import product


ROOT = 'reports/figures/Eksperimenter/2/testusik'

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9-fortræning',
            '3D-EMGP-lokalt': 'Lokalt',
            '3D-EMGP-globalt': 'Globalt',
            '3D-EMGP-begge': 'Begge'
            }

def sanity_check_group_df(group_df):
    assert len(group_df['seed'].unique()) == 32
    fortræer = group_df['fortræningsudgave'].unique()
    datamængder = group_df['eftertræningsmængde'].unique()
    for fortræ, datamængde in product(fortræer, datamængder):
        idxs = group_df['fortræningsudgave'] == fortræ
        idxs = (idxs) & (group_df['eftertræningsmængde'] == datamængde)
        if len(group_df[idxs]) != 32:
            print(f"fortræ = {fortræ}, datamængde = {datamængde}  er ikke ok. Den har {len(group_df[idxs])} frø!")
            duplicates = group_df[idxs]['seed'].value_counts()
            duplicates = duplicates[duplicates > 1].index
            print("Gengangere i serien:", duplicates)
            print("\n")
        else:
            print(f"fortræ = {fortræ}, datamængde = {datamængde}  er ok")
            print("\n")
        # assert len(group_df[idxs]) == 33


groups = ['eksp2_0']
for group in tqdm(groups):
    group_df = viz0.get_group_df(group)
    fortræningsudgaver, temperaturer, seeds = viz0.get_loop_params_group_df(group_df)
    sanity_check_group_df(group_df)

    # runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)

    # seeds = random.sample(list(seeds), k=4)
    # print(f"seeds = {seeds}")
    seeds = [158.0, 282.0, 332.0, 340.0]


    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)
    for temperatur in temperaturer:
        ncols = 2
        # nrows = len(seeds) // ncols + (1 if len(seeds) % ncols != 0 else 0)
        nrows = 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(10 * ncols, 6 * nrows))
        axs = axs.ravel()

        for idx, seed in enumerate(seeds):
            ax = axs[idx]
            row = idx // ncols
            col = idx % ncols
            # ax = axs[row, col]

            i = 0
            for fortræningsudgave in fortræningsudgaver:
                df = copy.deepcopy(group_df)
                df = df[df['fortræningsudgave'] == fortræningsudgave]
                df = df[df['seed'] == seed]
                df = df[df['temperatur'] == temperatur]

                # label = LABELLER[fortræningsudgave]
                label = viz0.FORT_LABELLER[fortræningsudgave]
                farve = viz0.FARVEOPSLAG[fortræningsudgave]
                ax.scatter(df["eftertræningsmængde"], df[f"test_loss_mean"], label=label, color=farve)
                ax.fill_between(df["eftertræningsmængde"], df[f"test_loss_lower"], df[f"test_loss_upper"],
                                color=farve, alpha=0.3)
                i += 1
            ax.set_xlabel("Datamængde", fontsize=22)
            ax.set_ylabel("MAE", fontsize=22)
            # ax.set_yscale("log")
            if idx == 0:
                ax.legend(fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=26)
            ax.tick_params(axis='both', which='minor', labelsize=13)

            x_ticks = group_df[group_df['seed'] == seed]['eftertræningsmængde'].unique()
            ax.set_xticks(x_ticks)

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
