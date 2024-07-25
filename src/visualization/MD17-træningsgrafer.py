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


def plot_lr(dfs: dict):
    w = 9
    golden_ratio = (5 ** .5 - 1) / 2
    h = w*golden_ratio
    fig, axs = plt.subplots(1, 2, figsize=(w*2, h))
    x_col = 'trainer/global_step'
    col = 'lr-AdamW'
    for i, (predicted_attribute, df) in enumerate(dfs.items()):
        # farve = far.corporate_red
        farve = viz0.predicted_attribute_to_background[predicted_attribute]
        ax = axs[i]
        e_p_s = df['epoch'].max() / df[x_col].max()
        data = df[[x_col, col]].dropna(how='any')
        x, y = data[x_col] * e_p_s, data[col]
        label = predicted_attribute_to_title[predicted_attribute]

        # Plot data
        ax.plot(x, y, color=farve, label=label, linewidth=4)



        # ændr baggrundsfarven
        for key in ['top', 'bottom', 'left', 'right']:
            ax.spines[key].set_color(viz0.predicted_attribute_to_background[predicted_attribute])
            ax.spines[key].set_linewidth(4)

        # Generel plot indstillinger
        ax.set_xlabel('Epoke', fontsize=22)
        ax.set_ylabel('Læringsrate', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        ax.grid(True)  # Tilføj grid for bedre læsbarhed
        ax.set_yscale('log')

        def custom_formatter(x, pos):
            return f'{x:.1e}'

        y_ticks = np.geomspace(df[col].min(), df[col].max(), num=5)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

        ax.minorticks_off()

        ax.legend(fontsize=28)

    plt.tight_layout()

    plt.savefig(os.path.join(ROOT, f"lersch.jpg"))
    plt.savefig(os.path.join(ROOT, f"lersch.pdf"))
    # plt.show()
    plt.close()


def plot(dfs: dict):
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))  # Øget figurstørrelse for bedre plads
    x_col = 'trainer/global_step'
    titles = {
        'force': [
            r'$\mathcal{L}_{\text{kræft}}$ på $\mathcal{D}_{\text{E-Tr}}$',
            r'$\mathcal{L}_{\text{kræft}}$ på $\mathcal{D}_{\text{E-Va}}$',
        ],
        'energy': [
            r'$\mathcal{L}_{\text{energi}}$ på $\mathcal{D}_{\text{E-Tr}}$',
            r'$\mathcal{L}_{\text{energi}}$ på $\mathcal{D}_{\text{E-Va}}$',
        ]
    }
    titles = {
        'force': [
            r'$\mathcal{L}_{\text{kræft}}$ på Efter-træn',
            r'$\mathcal{L}_{\text{kræft}}$ på Efter-val',
        ],
        'energy': [
            r'$\mathcal{L}_{\text{energi}}$ på Efter-træn',
            r'$\mathcal{L}_{\text{energi}}$ på Efter-val',
        ]
    }
    cols = {ds: [f'{task}_{ds}_loss' for task in ['train', 'val']] for ds in ['energy', 'force']}

    for i, (predicted_attribute, df) in enumerate(dfs.items()):
        # farve = far.corporate_red
        farve = viz0.predicted_attribute_to_background[predicted_attribute]
        for j, col in enumerate(cols[predicted_attribute]):
            ax = axs[i, j]
            for key in ['top', 'bottom', 'left', 'right']:
                ax.spines[key].set_color(viz0.predicted_attribute_to_background[predicted_attribute])
                ax.spines[key].set_linewidth(4)
            e_p_s = df['epoch'].max() / df[x_col].max()
            data = df[[x_col, col]].dropna(how='any')
            label = predicted_attribute_to_title[predicted_attribute]
            ax.plot(data[x_col]*e_p_s, data[col], color=farve, label=label, linewidth=4)
            ax.set_title(titles[predicted_attribute][j], fontsize=23)  # Tilføjelse af titler til subplots
            ax.set_xlabel('Epoke', fontsize=24)
            ax.set_ylabel('MSE', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=22)
            ax.tick_params(axis='both', which='minor', labelsize=20)
            ax.grid(True)  # Tilføj grid for bedre læsbarhed
            ax.set_yscale('log')
            if j == 0:
                ax.legend(fontsize=28)  # Tilføj legende kun én gang for at undgå overlap

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.suptitle('Træningsmetrikker over epoker', fontsize=28)  # Tilføj hovedtitel
    # plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    # plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))
    # plt.legend()
    plt.savefig(os.path.join(ROOT, "MD17træningskurve.jpg"))
    plt.savefig(os.path.join(ROOT, "MD17træningskurve.pdf"))
    plt.close()


def get_history(predicted_attribute, group):
    cache_path = os.path.join(CACHE, f'{predicted_attribute}.pickle')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        return cache['df']
    else:
        os.makedirs(CACHE, exist_ok=True)
        runs = wandb.Api().runs("afhandling")
        runs_in_group, _, _, _, _ = viz0.get_loops_params(group, runs)
        run = random.choice(runs_in_group)
        df = run.history(samples=10000)
        cache = {
            'df': df
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        return df


ROOT = os.path.join('reports/figures/Eksperimenter/MD17eftertræn')
CACHE = os.path.join('reports/cache/md17eftertræn')

groups = {
    'energy': 'eksp2-md17_2',
    'force': 'eksp2-force_0',
}
predicted_attribute_to_title = {
    'force': 'Kræfter',
    'energy': 'Energi'
}
# runs = wandb.Api().runs("afhandling")

dfs = {}
for predicted_attribute, group in groups.items():
    # grupperod = os.path.join(ROOT, group)
    # os.makedirs(grupperod, exist_ok=True)
    os.makedirs(ROOT, exist_ok=True)
    df = get_history(predicted_attribute, group)
    dfs[predicted_attribute] = df

    # runs_in_group, _, _, _, _ = viz0.get_loops_params(group, runs)
    # run = random.choice(runs_in_group)
    # dfs[predicted_attribute] = run.history(samples=10000)
# plot_faser(dfs)
plot(dfs)
plot_lr(dfs)