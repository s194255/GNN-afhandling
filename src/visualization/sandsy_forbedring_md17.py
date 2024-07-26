import copy

import pandas as pd

from src.visualization import viz0
import numpy as np
import os
import pickle
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import seaborn as sns
import src.visualization.farver as far

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)

def getGM(group_df, idxs, fortræ, col):
    idxs2 = (idxs) & (group_df['fortræningsudgave'] == fortræ)
    X = group_df[idxs2][[col]]
    param_grid = {
        "n_components": range(1, 2)
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(X)
    return grid_search

def plot_sandsy_forbedring(forb_df: pd.DataFrame):
    w = 10
    golden_ratio = (5 ** .5 - 1) / 2
    h = w * golden_ratio
    fig, ax = plt.subplots(figsize=(w, h))

    nøgle_to_xytext = {
        'force': (-35, 10),
        'energy': (0, -20)
    }
    nøgle_to_labels = {
        'force': 'Kræfter',
        'energy': 'Energi'
    }

    # Plot dataen
    for nøgle in ['energy', 'force']:
        idxs = forb_df['nøgle'] == nøgle
        color = viz0.predicted_attribute_to_background[nøgle]
        label = nøgle_to_labels[nøgle]
        ax.plot(forb_df[idxs]['eftertræningsmængde'], forb_df[idxs]['sandsy_forbedring'], color=color, linewidth=4, marker='o',
                markersize=10, label=label)

        # Annoter punkterne med deres faktiske værdier
        xytext = nøgle_to_xytext[nøgle]
        for x, y in zip(forb_df[idxs]['eftertræningsmængde'], forb_df[idxs]['sandsy_forbedring']):
            ax.annotate(f'{y:.2%}', xy=(x, y), xytext=xytext, textcoords='offset points', fontsize=12, color=color)

    # Tilføj titler og labels
    # ax.set_title('Sandsynlighed for forbedring vs. Datamængde', fontsize=16)
    ax.set_xlabel(r'Datamængde ($N_{træn}$)', fontsize=22)
    ax.set_ylabel(r'$P(t_m < t_u)$', fontsize=22)

    # Tilføj grid
    ax.grid(True, linestyle='--', alpha=0.6)

    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], ylim[1] + 0.05])

    xlim = ax.get_xlim()
    ax.set_xlim([xlim[0]-10, xlim[1]])

    # Tilføj legend
    ax.legend(fontsize=15)

    # Tilføj en formatter til y-aksen for at vise procenter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Sæt xticks baseret på de unikke værdier i 'eftertræningsmængde' kolonnen
    unique_xticks = forb_df['eftertræningsmængde'].unique()
    ax.set_xticks(unique_xticks)

    ax.tick_params(axis='both', which='both', labelsize=16)

    # Gem figuren i både jpg og pdf format
    filename = 'sandsyforb_vs_datamængde'
    plt.savefig(os.path.join(group_root, f"{filename}.jpg"))
    plt.savefig(os.path.join(group_root, f"{filename}.pdf"))
    plt.close()


def skab_sandsy_forbedring(group_dfs: dict):
    forb_df = pd.DataFrame(data={'eftertræningsmængde': [], 'sandsy_forbedring': [], 'nøgle': []})
    for nøgle, group_df in group_dfs.items():
        ems = sorted(group_df['eftertræningsmængde'].unique())
        plot_em = ems[-1]
        n_samples = 10**7
        gmm_dict = {}
        col = f'test_{nøgle}_loss'
        for em in ems:
            print(f"datamængde = {em}")
            idxs = group_df['eftertræningsmængde'] == em
            samples = {}
            gmm_dict[em] = {}
            for fortræ in ['uden', '3D-EMGP-begge']:
                gs = getGM(group_df, idxs, fortræ, col)
                print(f"{fortræ} bedste paramer = {gs.best_params_}")
                sample = gs.best_estimator_.sample(n_samples=n_samples)[0].squeeze(1)
                samples[fortræ] = sample
                gmm_dict[em][fortræ] = gs.best_estimator_

            diff = samples['uden'] - samples['3D-EMGP-begge']
            sandsy_forbedring = np.mean(diff > 0)
            n = len(group_df[idxs])
            print(f"sandsy = {sandsy_forbedring}")
            print("\n")
            række = {
                'eftertræningsmængde': [em],
                'sandsy_forbedring': [sandsy_forbedring],
                'nøgle': [nøgle]
            }
            forb_df = pd.concat([forb_df, pd.DataFrame(data=række)], ignore_index=True)
            if plot_em == em:
                pass

    plot_sandsy_forbedring(forb_df)

ROOT = 'reports/figures/Eksperimenter/sandsy_forbedring'


# groups = ['eksp2_0', 'sandsyForbedring_0']
groups = [
    {
        'name': 'md17_main',
        'force': 'eksp2-force_0',
        'energy': 'eksp2-md17_2'
    }
]
for group in groups:
    group_root = os.path.join(ROOT, group['name'])
    os.makedirs(group_root, exist_ok=True)
    group_dfs = {nøgle: viz0.get_group_df(group[nøgle]) for nøgle in ['force', 'energy']}
    skab_sandsy_forbedring(group_dfs)