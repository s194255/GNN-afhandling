import shutil
import os
import matplotlib.pyplot as plt
# from src.visualization.farver import farver
from matplotlib.ticker import ScalarFormatter, MultipleLocator, AutoMinorLocator
import src.visualization.farver as far
from tqdm import tqdm
from src.visualization import viz0
import numpy as np
import pandas as pd

TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet rygrad"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9-fortræning',
            '3D-EMGP-lokalt': 'Lokalt',
            '3D-EMGP-globalt': 'Globalt',
            '3D-EMGP-begge': 'Begge'
            }

FIGNAVN = 'trænusik2'
ROOT = os.path.join('reports/figures/Eksperimenter/2', FIGNAVN)

farver = [far.corporate_red, far.blue, far.navy_blue, far.bright_green, far.orange, far.yellow]
stjerner = viz0.get_stjerner()
print(stjerner)

def plot_kernel_baseline(ax, x_values, x, farve, df2=None):
    kernel = viz0.kernel_baseline()
    # x = np.linspace(30, 500, 1000)
    y = kernel(x_values)
    if df2 is not None:
        y = 100 * y / df2['test_loss_mean']
    ax.scatter(x, y, color=farve, marker="d", label="kernel baseline", s=80,
               edgecolor=far.black)

def plot_normaliseret_dobbelt(df1, fortræningsudgaver):
    # Opsætning for søjlerne
    x_values = df1['eftertræningsmængde'].unique()
    x_values.sort()
    # x_values = x_values.astype(int)
    # fortræningsudgaver = df['fortræningsudgave'].unique()
    num_models = len(fortræningsudgaver)

    df2 = df1[['eftertræningsmængde', 'test_loss_mean']][df1['fortræningsudgave'] == 'uden']
    df2 = df2.groupby(by='eftertræningsmængde').mean().reset_index()

    # Merge de to dataframes på 'eftertræningsmængde'
    df1 = pd.merge(df1, df2, on='eftertræningsmængde', suffixes=('', '_df2'))

    # Normaliser 'test_loss' i df1 med 'test_loss' fra df2
    df1['normalized_test_loss'] = 100 * df1['test_loss_mean'] / df1['test_loss_mean_df2']


    bar_width = 0.15
    x = np.arange(len(x_values))

    # Opret figuren og aksern
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
    cols = ['test_loss_mean', 'normalized_test_loss']
    for c, col in enumerate(cols):
        ax = axs[c]

        # Plot søjlerne og prikkerne
        for i in range(num_models):

            fortræningsudgave = fortræningsudgaver[i]
            målinger = df1[df1['fortræningsudgave'] == fortræningsudgave][['eftertræningsmængde', col]]
            søjlehøjde = målinger.groupby('eftertræningsmængde').mean().reset_index()[col]

            if len(søjlehøjde) != len(x_values):
                continue
            bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, søjlehøjde,
                          bar_width, color=farver[i], alpha=0.85)
            for j in range(len(x_values)):
                prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]][col]
                n2 = len(prikker)
                label = LABELLER[fortræningsudgave] if j==0 else None
                ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker,
                           color=farver[i], label=label, marker='o', edgecolor='black', alpha=1.0)

        if col == 'normalized_test_loss':
            plot_kernel_baseline(ax, x_values, x, farver[i+1], df2)
            ax.yaxis.set_major_locator(MultipleLocator(25))  # Hovedticks hver 1 enhed
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Mindre ticks hver halve enhed
            titel = 'Normaliseret'
            ylabel = 'Normaliseret MAE'

        elif col == 'test_loss_mean':
            plot_kernel_baseline(ax, x_values, x, farver[i + 1])
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            titel = 'Rede tal'
            ylabel = 'MAE'

        # Tilpasning af akserne og labels
        ax.set_xlabel('Datamængde', fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(titel, fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(x_values.astype(int))
        ax.legend(fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=19)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        # plt.tight_layout()
        # ax.grid()

    fig.suptitle("Optøet", fontsize=25)
    plt.tight_layout(rect=[0, 0, 1, 0.999999999])
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.pdf"))
    plt.close()


def plot_normalisere_enkelt(df1, fortræningsudgaver):
    # Opsætning for søjlerne
    x_values = df1['eftertræningsmængde'].unique()
    x_values.sort()
    # x_values = x_values.astype(int)
    # fortræningsudgaver = df['fortræningsudgave'].unique()
    num_models = len(fortræningsudgaver)

    df2 = df1[['eftertræningsmængde', 'test_loss_mean']][df1['fortræningsudgave'] == 'uden']
    df2 = df2.groupby(by='eftertræningsmængde').mean().reset_index()

    # Merge de to dataframes på 'eftertræningsmængde'
    df1 = pd.merge(df1, df2, on='eftertræningsmængde', suffixes=('', '_df2'))

    # Normaliser 'test_loss' i df1 med 'test_loss' fra df2
    df1['normalized_test_loss'] = 100 * df1['test_loss_mean'] / df1['test_loss_mean_df2']

    bar_width = 0.15
    x = np.arange(len(x_values))

    # Opret figuren og aksern
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
    cols = ['normalized_test_loss']
    for c, col in enumerate(cols):
        ax = axs

        # Plot søjlerne og prikkerne
        for i in range(num_models):

            fortræningsudgave = fortræningsudgaver[i]
            målinger = df1[df1['fortræningsudgave'] == fortræningsudgave][['eftertræningsmængde', col]]
            søjlehøjde = målinger.groupby('eftertræningsmængde').mean().reset_index()[col]

            if len(søjlehøjde) != len(x_values):
                continue
            bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, søjlehøjde,
                          bar_width, color=farver[i], alpha=0.85)
            for j in range(len(x_values)):
                prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]][col]
                n2 = len(prikker)
                label = LABELLER[fortræningsudgave] if j == 0 else None
                ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker,
                           color=farver[i], label=label, marker='o', edgecolor='black', alpha=1.0)

        if col == 'normalized_test_loss':
            plot_kernel_baseline(ax, x_values, x, farver[i + 1], df2)
            ax.yaxis.set_major_locator(MultipleLocator(25))  # Hovedticks hver 1 enhed
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Mindre ticks hver halve enhed
            titel = TITLER[temperatur]
            ylabel = 'Normaliseret MAE'

        elif col == 'test_loss_mean':
            plot_kernel_baseline(ax, x_values, x, farver[i + 1])
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            titel = TITLER[temperatur]
            ylabel = 'MAE'

        # Tilpasning af akserne og labels
        ax.set_xlabel('Datamængde', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(titel, fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(x_values.astype(int))
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        # plt.tight_layout()
        # ax.grid()

    # fig.suptitle("Optøet", fontsize=25)
    # plt.tight_layout(rect=[0, 0, 1, 0.999999999])
    plt.tight_layout()
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.pdf"))
    plt.close()

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    if stjerner != None:
        if group not in [f'eksp2_{udvalgt}' for udvalgt in stjerner]:
            continue
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    eftertræningsmængder = viz0.get_eftertræningsmængder(group, runs)
    assert len(temperaturer) == 1
    temperatur = list(temperaturer)[0]
    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)

    runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave=None, seed=None), runs_in_group))
    df = viz0.get_df(runs_filtered)

    # plot_normaliseret_dobbelt(df, fortræningsudgaver)
    plot_normalisere_enkelt(df, fortræningsudgaver)