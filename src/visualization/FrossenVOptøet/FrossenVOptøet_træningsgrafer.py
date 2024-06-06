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


def plot(dfs: dict):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # Øget figurstørrelse for bedre plads
    x_col = 'trainer/global_step'
    cols = ['lr-AdamW', 'train_loss', 'val_loss']
    titles = ['Læringsrate', 'Træningstab', 'Valideringstab']  # Tilføjet titler for subplots

    for i, (temperatur, df) in enumerate(dfs.items()):
        farve = farveopslag.get(temperatur, 'blue')  # Standardfarve hvis temperatur ikke findes i farveopslag
        for j, col in enumerate(cols):
            ax = axs[i, j]
            e_p_s = df['epoch'].max() / df[x_col].max()
            data = df[[x_col, col]].dropna(how='any')
            ax.plot(data[x_col]*e_p_s, data[col], color=farve, label=f'{temperatur}')
            ax.set_title(titles[j], fontsize=18)  # Tilføjelse af titler til subplots
            ax.set_xlabel('Epoke', fontsize=16)
            ax.set_ylabel(col.replace('_', ' ').title(), fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.grid(True)  # Tilføj grid for bedre læsbarhed
            ax.set_yscale('log')
            if j == 0:
                ax.legend(fontsize=20)  # Tilføj legende kun én gang for at undgå overlap

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.suptitle('Træningsmetrikker over globale skridt', fontsize=23)  # Tilføj hovedtitel
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
    dfs[temperatur] = run.history()
plot(dfs)