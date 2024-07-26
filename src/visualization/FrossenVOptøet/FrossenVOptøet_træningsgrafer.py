import shutil
import os
import matplotlib.pyplot as plt
import pandas as pd
# from src.visualization.farver import farver
from matplotlib.ticker import ScalarFormatter, FuncFormatter
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
        label = LABELLER[temperatur]

        # Plot data
        ax.plot(x, y, color=farve, label=label, linewidth=4)

        # Generel plot indstillinger
        ax.set_xlabel('Epoke', fontsize=16)
        ax.set_ylabel('Læringsrate', fontsize=16)
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

        def custom_formatter(x, pos):
            return f'{x:.1e}'

        y_ticks = np.geomspace(df[col].min(), df[col].max(), num=5)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

        ax.minorticks_off()

        ax.legend(fontsize=28)

    plt.tight_layout()

    # fignavn = 'faser'
    # rod = os.path.join('reports/figures/Eksperimenter/FrossenVOptøet', fignavn)
    #
    # if os.path.exists(rod):
    #     shutil.rmtree(rod)
    # os.makedirs(rod)

    plt.savefig(os.path.join(grupperod, f"faser.jpg"))
    plt.savefig(os.path.join(grupperod, f"faser.pdf"))
    # plt.show()
    plt.close()



def plot(dfs: dict):
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))  # Øget figurstørrelse for bedre plads
    x_col = 'trainer/global_step'
    # cols = ['lr-AdamW', 'train_loss', 'val_loss']
    cols = ['train_loss', 'val_loss']
    titles = [
        'Træningstab',
        'Valideringstab'
    ]

    titles = [
        r'$\mathcal{L}_{\mathfrak{p}}$ på Efter-træn',
        r'$\mathcal{L}_{\mathfrak{p}}$ på Efter-val'
    ]

    for i, (temperatur, df) in enumerate(dfs.items()):
        farve = farveopslag.get(temperatur, 'blue')  # Standardfarve hvis temperatur ikke findes i farveopslag
        for j, col in enumerate(cols):
            ax = axs[i, j]
            e_p_s = df['epoch'].max() / df[x_col].max()
            data = df[[x_col, col]].dropna(how='any')
            label = LABELLER[temperatur]
            ax.plot(data[x_col]*e_p_s, data[col], color=farve, label=label)
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
    plt.subplots_adjust(top=0.90)
    fig.suptitle('Træningsmetrikker over epoker', fontsize=28)  # Tilføj hovedtitel
    plt.savefig(os.path.join(grupperod, f"træningskurve.jpg"))
    plt.savefig(os.path.join(grupperod, f"træningskurve.pdf"))
    plt.close()



TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet rygrad"}

LABELLER = {
    'frossen': 'Frossen',
    'optøet': 'Optøet'
}
temperaturer = ['frossen', 'optøet']

farveopslag = {
    'optøet': far.corporate_red,
    'frossen': far.blue
}

ROOT = os.path.join('reports/figures/Eksperimenter/FrossenVOptøet')

groups = ['eksp4_3']
runs = wandb.Api().runs("afhandling")

for group in groups:
    grupperod = os.path.join(ROOT, group)
    os.makedirs(grupperod, exist_ok=True)

    runs_in_group, fortræningsudgaver, _, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    temper = ['frossen', 'optøet']
    dfs = {}
    for temp in temper:
        temp_runs = list(filter(lambda run: viz0.get_temperatur(run) == temp, runs_in_group))
        run = random.choice(temp_runs)
        dfs[temp] = run.history(samples=10000)
    plot_faser(dfs)
    plot(dfs)