import shutil
import os
import matplotlib.pyplot as plt
import pandas as pd
# from src.visualization.farver import farver
from matplotlib.ticker import ScalarFormatter
import src.visualization.farver as far
from tqdm import tqdm
from src.visualization import viz0
import numpy as np
import pickle
import wandb
import random


def plot_faser(dfs: dict):
    fig, axs = plt.subplots(1, 2, figsize=(18, 4))
    x_col = 'trainer/global_step'
    col = 'lr-AdamW'
    x_skiller = {'optøet': 100,
                 'frossen': 1}
    for i, (temperatur, df) in enumerate(dfs.items()):
        farve = farveopslag[temperatur]
        ax = axs[i]
        e_p_s = df['epoch'].max() / df[x_col].max()
        data = df[[x_col, col]].dropna(how='any')
        x, y = data[x_col] * e_p_s, data[col]


        # Plot data
        ax.plot(x, y, color=farve, label=f'{temperatur}', linewidth=4)

        # Generel plot indstillinger
        ax.set_xlabel('Epoke', fontsize=16)
        ax.set_ylabel('MAE', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.grid(True)  # Tilføj grid for bedre læsbarhed
        ax.set_yscale('log')

        x_skille = x_skiller[temperatur]

        # Farv baggrunden med gennemsigtighed
        ax.axvspan(0, x_skille, facecolor='#79238E', alpha=0.2, zorder=0)
        ax.axvspan(x_skille, max(x), facecolor='#FC7634', alpha=0.2, zorder=0)

        # Tilføj tykke kanter omkring de markerede områder
        # Venstre område
        left_patch = plt.Rectangle((0, ax.get_ylim()[0]), x_skille, ax.get_ylim()[1] - ax.get_ylim()[0],
                                   edgecolor='#5A1E63', facecolor='none', linewidth=3, zorder=1)
        ax.add_patch(left_patch)

        # Højre område
        right_patch = plt.Rectangle((x_skille, ax.get_ylim()[0]), max(x) - x_skille, ax.get_ylim()[1] - ax.get_ylim()[0],
                                    edgecolor='#B24E36', facecolor='none', linewidth=3, zorder=1)
        ax.add_patch(right_patch)

        if temperatur == 'optøet':
            y_cord = np.geomspace(ax.get_ylim()[0], ax.get_ylim()[1], 3)[1]
            ax.text(x_skille / 2, y_cord, 'fase 1', ha='center', va='center', fontsize=20, color='#5A1E63',
                    zorder=2)
            ax.text(x_skille + (max(x) - x_skille) / 2, y_cord, 'fase 2', ha='center', va='center',
                    fontsize=20, color='#B24E36', zorder=2)
        elif temperatur == 'frossen':
            y_cord = np.geomspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)[75]
            ax.text(x_skille + (max(x) - x_skille) / 2, y_cord, 'fase 2', ha='center', va='center',
                    fontsize=20, color='#B24E36', zorder=2)


        ax.legend(fontsize=28)

    plt.tight_layout()

    fignavn = 'faser'
    rod = os.path.join('reports/figures/Eksperimenter/FrossenVOptøet', fignavn)

    if os.path.exists(rod):
        shutil.rmtree(rod)
    os.makedirs(rod)

    plt.savefig(os.path.join(rod, f"{fignavn}.jpg"))
    plt.savefig(os.path.join(rod, f"{fignavn}.pdf"))
    # plt.show()
    plt.close()



def plot(dfs: dict):
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))  # Øget figurstørrelse for bedre plads
    x_col = 'trainer/global_step'
    # cols = ['lr-AdamW', 'train_loss', 'val_loss']
    cols = ['train_loss', 'val_loss']
    titles = ['Træningstab', 'Valideringstab']  # Tilføjet titler for subplots

    for i, (temperatur, df) in enumerate(dfs.items()):
        farve = farveopslag.get(temperatur, 'blue')  # Standardfarve hvis temperatur ikke findes i farveopslag
        for j, col in enumerate(cols):
            ax = axs[i, j]
            e_p_s = df['epoch'].max() / df[x_col].max()
            data = df[[x_col, col]].dropna(how='any')
            ax.plot(data[x_col]*e_p_s, data[col], color=farve, label=f'{temperatur}')
            ax.set_title(titles[j], fontsize=23)  # Tilføjelse af titler til subplots
            ax.set_xlabel('Epoke', fontsize=16)
            ax.set_ylabel('MAE', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.grid(True)  # Tilføj grid for bedre læsbarhed
            ax.set_yscale('log')
            if j == 0:
                ax.legend(fontsize=28)  # Tilføj legende kun én gang for at undgå overlap

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Træningsmetrikker over epoker', fontsize=28)  # Tilføj hovedtitel
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))
    plt.legend()
    plt.close()



TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet rygrad"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9 fortræning',
            '3D-EMGP-lokalt': '3D-EMGP kun lokalt',
            '3D-EMGP-globalt': '3D-EMGP kun globalt',
            '3D-EMGP-begge': '3D-EMGP'
            }
temperaturer = ['frossen', 'optøet']


FIGNAVN = 'træningskurve'
ROOT = os.path.join('reports/figures/Eksperimenter/FrossenVOptøet', FIGNAVN)

farveopslag = {
    'optøet': far.corporate_red,
    'frossen': far.blue
}
stjerner = {
    'optøet': 'eksp2_83',
    'frossen': 'eksp2_88'
}


if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

runs = wandb.Api().runs("afhandling")
runs = list(filter(lambda w: viz0.is_suitable(w, 'eksp2'), runs))
df = None
kørsel_path = os.path.join(ROOT)
os.makedirs(kørsel_path)


dfs = {}
for temperatur in temperaturer:
    group = stjerner[temperatur]
    runs_in_group, fortræningsudgaver, temperaturer_lp, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    runs_in_group = list(filter(lambda run: viz0.get_eftertræningsmængde(run) == 500, runs_in_group))
    run = random.choice(runs_in_group)
    print(temperatur, run.id)
    dfs[temperatur] = run.history()
plot(dfs)
plot_faser(dfs)