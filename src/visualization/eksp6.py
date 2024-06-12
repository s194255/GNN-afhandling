print("importerer ...")
import shutil
import os
import matplotlib.pyplot as plt
from src.visualization.farver import farver
# from src.visualization import viz0
from src.visualization.viz0 import get_group_df, get_loop_params_group_df
import pandas as pd
from tqdm import tqdm
import copy
print("færdig med at importere ...")

#%%

def plot(group_df: pd.DataFrame):
    for temperatur in temperaturer:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # fig, ax = plt.figure(figsize=(10,6))

        i = 0
        for fortræningsudgave in fortræningsudgaver:
            df = copy.deepcopy(group_df)
            df = df[df['fortræningsudgave'] == fortræningsudgave]
            # df = df[df['seed'] == seed]
            df = df[df['temperatur'] == temperatur]

            df = df[['eftertræningsmængde', 'test_loss_mean']]
            df = df.groupby(by='eftertræningsmængde').mean().reset_index()

            label = LABELLER[fortræningsudgave]
            ax.plot(df["eftertræningsmængde"], df[f"test_loss_mean"], label=label, color=farver[i])
            # ax.fill_between(df["eftertræningsmængde"], df[f"test_loss_lower"], df[f"test_loss_upper"],
            #                 color=farver[i], alpha=0.3)
            i += 1
        ax.set_xlabel("Datamængde", fontsize=22)
        ax.set_ylabel("MAE", fontsize=22)
        ax.set_yscale("log")
        ax.legend(fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.tick_params(axis='both', which='minor', labelsize=13)

        plt.tight_layout()
        for ext in ['jpg', 'pdf']:
            plt.savefig(os.path.join(kørsel_path, f"{temperatur}_konvergens.{ext}"))
        plt.close()


ROOT = 'reports/figures/Eksperimenter/6'

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9-fortræning',
            '3D-EMGP-lokalt': 'Lokalt',
            '3D-EMGP-globalt': 'Globalt',
            '3D-EMGP-begge': 'Begge'
            }

groups = ['eksp6_1']
for group in tqdm(groups):
    group_df = get_group_df(group)
    fortræningsudgaver, temperaturer, seeds = get_loop_params_group_df(group_df)

    # runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    # seeds = random.sample(list(seeds), k=4)
    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)
    plot(group_df)
