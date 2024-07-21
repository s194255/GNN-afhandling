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

def getGM(group_df, idxs, fortræ):
    idxs2 = (idxs) & (group_df['fortræningsudgave'] == fortræ)
    X = group_df[idxs2][['test_loss_mean']]
    param_grid = {
        "n_components": range(1, 7)
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )
    grid_search.fit(X)
    return grid_search

def plot_hist2(group_df, ems, gmm_dict):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(3/5*30, 2*6))
    axs = axs.ravel()
    for j, em in enumerate(ems):
        ax = axs[j]
        level = 0
        idxs = group_df['eftertræningsmængde'] == em
        linspacesart = group_df[idxs]['test_loss_mean'].min()-20
        for i, fortræ in enumerate(['uden', '3D-EMGP-begge']):
            gmm = gmm_dict[em][fortræ]
            color = viz0.FARVEOPSLAG[fortræ]
            idxs2 = (idxs) & (group_df['fortræningsudgave'] == fortræ)
            data = group_df[idxs2]['test_loss_mean']
            x = np.linspace(linspacesart, max(data), 1000).reshape(-1, 1)
            logprob = gmm.score_samples(x)
            pdf = np.exp(logprob)

            # ax.hist(data, bins=20, density=True, alpha=0.8, color=color, label='observeret')
            hist, bins = np.histogram(data, bins=20, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Plot histogrammet
            barlabel = viz0.FORT_LABELLER[fortræ]
            ax.bar(bin_centers, hist, width=np.diff(bins), align='center', alpha=0.8, color=color, label=barlabel,
                   bottom=level)
            plot_label = 'GMM-fit' if (i==0) else None
            ax.plot(x, pdf+level, linewidth=4, label=plot_label, color=far.black)
            hævelevel = max(np.max(hist), np.max(pdf))
            level += 1.2*hævelevel
        ax.set_xlabel('MAE', fontsize=18)
        ax.set_ylabel('Tæthed', fontsize=18)
        title = r'$N_{træn}$ = ' + f'{int(em)}'
        ax.set_title(title, fontsize=24)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        if j == 4:
            ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)

    for k in range(len(ems), len(axs)):
        fig.delaxes(axs[k])

    plt.tight_layout()
    plt.savefig(os.path.join(group_root, "hist_gmm.jpg"))
    plt.savefig(os.path.join(group_root, "hist_gmm.pdf"))
    plt.close()

def plot_hist(group_df, idxs, gmm_dict):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
    for i, fortræ in enumerate(['uden', '3D-EMGP-begge']):
        ax = axs[i]
        gmm = gmm_dict[fortræ]
        color = viz0.FARVEOPSLAG[fortræ]
        idxs2 = (idxs) & (group_df['fortræningsudgave'] == fortræ)
        data = group_df[idxs2]['test_loss_mean']
        x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)
        logprob = gmm.score_samples(x)
        pdf = np.exp(logprob)

        ax.hist(data, bins=20, density=True, alpha=0.8, color=color, label='observeret')
        ax.plot(x, pdf, linewidth=4, label='GMM-fit', color=far.black)
        ax.set_title(viz0.FORT_LABELLER[fortræ], fontsize=22)
        ax.set_xlabel('MAE', fontsize=18)
        ax.set_ylabel('Tæthed', fontsize=18)
        ax.legend(fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig(os.path.join(group_root, "hist_gmm.jpg"))
    plt.savefig(os.path.join(group_root, "hist_gmm.pdf"))
    plt.close()

def plot_sandsy_forbedring(forb_df: pd.DataFrame):
    w = 10
    golden_ratio = (5 ** .5 - 1) / 2
    h = w * golden_ratio
    fig, ax = plt.subplots(figsize=(w, h))

    # Definer farve
    color = viz0.FARVEOPSLAG['3D-EMGP-begge']

    # Plot dataen
    ax.plot(forb_df['eftertræningsmængde'], forb_df['sandsy_forbedring'], color=color, linewidth=4, marker='o',
            markersize=10, label='Sandsynlighed for forbedring')

    # Tilføj titler og labels
    # ax.set_title('Sandsynlighed for forbedring vs. Datamængde', fontsize=16)
    ax.set_xlabel(r'Datamængde ($N_{træn}$)', fontsize=22)
    ax.set_ylabel('Sandsynlighed for forbedring', fontsize=22)

    # Tilføj grid
    ax.grid(True, linestyle='--', alpha=0.6)

    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], ylim[1] + 0.05])

    # Tilføj legend
    # ax.legend(fontsize=12)

    # Tilføj en formatter til y-aksen for at vise procenter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Sæt xticks baseret på de unikke værdier i 'eftertræningsmængde' kolonnen
    unique_xticks = forb_df['eftertræningsmængde'].unique()
    ax.set_xticks(unique_xticks)

    ax.tick_params(axis='both', which='both', labelsize=16)

    # Annoter punkterne med deres faktiske værdier
    for x, y in zip(forb_df['eftertræningsmængde'], forb_df['sandsy_forbedring']):
        ax.annotate(f'{y:.0%}', xy=(x, y), xytext=(-20, 10), textcoords='offset points', fontsize=12, color='black')

    # Gem figuren i både jpg og pdf format
    filename = 'sandsyforb_vs_datamængde'
    plt.savefig(os.path.join(group_root, f"{filename}.jpg"))
    plt.savefig(os.path.join(group_root, f"{filename}.pdf"))
    plt.close()


def skab_sandsy_forbedring(group_df: pd.DataFrame):
    ems = sorted(group_df['eftertræningsmængde'].unique())
    plot_em = ems[-1]
    n_samples = 10**7
    forb_df = pd.DataFrame(data={'eftertræningsmængde': [], 'sandsy_forbedring': []})
    gmm_dict = {}
    for em in ems:
        print(f"datamængde = {em}")
        idxs = group_df['eftertræningsmængde'] == em
        samples = {}
        gmm_dict[em] = {}
        for fortræ in ['uden', '3D-EMGP-begge']:
            gs = getGM(group_df, idxs, fortræ)
            print(f"{fortræ} bedste paramer = {gs.best_params_}")
            sample = gs.best_estimator_.sample(n_samples=n_samples)[0].squeeze(1)
            samples[fortræ] = sample
            gmm_dict[em][fortræ] = gs.best_estimator_

        diff = samples['uden'] - samples['3D-EMGP-begge']
        sandsy_forbedring = np.mean(diff > 0)
        n = len(group_df[idxs])
        print(f"sandsy = {sandsy_forbedring}")
        print(f"n = {n}")
        print("\n")
        række = {
            'eftertræningsmængde': [em],
            'sandsy_forbedring': [sandsy_forbedring]
        }
        forb_df = pd.concat([forb_df, pd.DataFrame(data=række)], ignore_index=True)
        if plot_em == em:
            pass
            # plot_violin2(group_df, idxs, samples)
            # plot_hist(group_df, idxs, gmm_dict)

    plot_hist2(group_df, ems, gmm_dict)
    plot_sandsy_forbedring(forb_df)

ROOT = 'reports/figures/Eksperimenter/sandsy_forbedring'


# groups = ['eksp2_0', 'sandsyForbedring_0']
groups = ['sandsyForbedring_0']
for group in groups:
    group_root = os.path.join(ROOT, group)
    os.makedirs(group_root, exist_ok=True)
    group_df = viz0.get_group_df(group)
    skab_sandsy_forbedring(group_df)