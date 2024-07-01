import shutil
import os
import matplotlib.pyplot as plt
# from src.visualization.farver import farver
# from matplotlib.ticker import ScalarFormatter, MultipleLocator, AutoMinorLocator
import src.visualization.farver as far
from tqdm import tqdm
from src.visualization import viz0
import numpy as np
import pandas as pd

TITLER = {'frossen': "Dyst (frossen)",
          'optøet': "Dyst"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9-fortræning',
            '3D-EMGP-lokalt': 'Lokalt',
            '3D-EMGP-globalt': 'Globalt',
            '3D-EMGP-begge': 'Begge'
            }
FARVEOPSLAG = {
    '3D-EMGP-lokalt': far.bright_green,
    '3D-EMGP-globalt': far.blue,
    '3D-EMGP-begge': far.navy_blue,
    'SelvvejledtQM9': far.orange,
    'uden': far.corporate_red,
}
fignavn = {
    'trad': 'trænusik2',
    'norm': 'trænusik2_normaliseret'
}
rod = lambda x: os.path.join('reports/figures/Eksperimenter/2', x)

# farver = [far.corporate_red, far.blue, far.navy_blue, far.bright_green, far.orange, far.yellow]


def plot_kernel_baseline(ax, x_values, x, farve, predicted_attribute):
    kernel = viz0.kernel_baseline(predicted_attribute)
    # x = np.linspace(30, 500, 1000)
    y = kernel(x_values)
    ax.scatter(x, y, color=farve, marker="d", label="kernel baseline", s=80,
               edgecolor=far.black)


def plot(df, fortræningsudgaver):
    # Opsætning for søjlerne
    x_values = df['eftertræningsmængde'].unique()
    x_values.sort()
    num_models = len(fortræningsudgaver)
    predicted_attribute = df['predicted_attribute'].unique()
    assert len(predicted_attribute) == 1
    predicted_attribute = predicted_attribute[0]


    bar_width = 0.15
    x = np.arange(len(x_values))

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot søjlerne og prikkerne
    for i in range(num_models):

        fortræningsudgave = fortræningsudgaver[i]
        målinger = df[df['fortræningsudgave'] == fortræningsudgave][['eftertræningsmængde', 'test_loss_mean']]
        søjlehøjde = målinger.groupby('eftertræningsmængde').mean().reset_index()['test_loss_mean']
        if len(søjlehøjde) != len(x_values):
            continue
        farve = FARVEOPSLAG[fortræningsudgave]
        bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, søjlehøjde,
                      bar_width, color=farve, alpha=0.85)
        for j in range(len(x_values)):
            prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]]['test_loss_mean']
            n2 = len(prikker)
            label = LABELLER[fortræningsudgave] if j==0 else None
            ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker,
                       color=farve, label=label, marker='o', edgecolor='black', alpha=1.0)

    plot_kernel_baseline(ax, x_values, x, far.yellow, predicted_attribute)

    # Tilpasning af akserne og labels
    ax.set_xlabel('Datamængde', fontsize=16)
    ax.set_ylabel('MAE', fontsize=16)
    ax.set_title(TITLER[temperatur], fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values.astype(int))
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    # ax.set_yscale("log")
    # ax.yaxis.set_minor_formatter(ScalarFormatter())
    # ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.tight_layout()

    kørsel_path = kørsel_paths['trad']
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{fignavn['trad']}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{fignavn['trad']}.pdf"))
    plt.close()

def plot_normalisere_enkelt(df1, fortræningsudgaver):
    # Opsætning for søjlerne
    x_values = df1['eftertræningsmængde'].unique()
    x_values.sort()
    # fortræningsudgaver = list(set(fortræningsudgaver) - {'uden'})
    fortræningsudgaver.remove('uden')
    num_models = len(fortræningsudgaver)

    df2 = df1[['eftertræningsmængde', 'test_loss_mean']][df1['fortræningsudgave'] == 'uden']
    df2 = df2.groupby(by='eftertræningsmængde').mean().reset_index()

    df1 = pd.merge(df1, df2, on='eftertræningsmængde', suffixes=('', '_df2'))

    # Normaliser 'test_loss' i df1 med 'test_loss' fra df2
    # df1['normalized_test_loss'] = 100 * df1['test_loss_mean'] / df1['test_loss_mean_df2']
    df1['normalized_test_loss'] = 100 * (df1['test_loss_mean'] - df1['test_loss_mean_df2']) / df1['test_loss_mean_df2']

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
            farve = FARVEOPSLAG[fortræningsudgave]
            bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, søjlehøjde, bar_width, color=farve,
                          alpha=0.85, zorder=2)
            for j in range(len(x_values)):
                prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]][col]
                n2 = len(prikker)
                label = LABELLER[fortræningsudgave] if j == 0 else None
                ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker, color=farve, label=label,
                           marker='o', edgecolor='black', alpha=1.0, zorder=3)

        if col == 'normalized_test_loss':
            # ax.yaxis.set_major_locator(MultipleLocator(25))  # Hovedticks hver 25 enheder
            # ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # Mindre ticks hver 5 enheder
            titel = f'%-vis reduktion ift. Ingen'
            ylabel = '%'

        # Tilpasning af akserne og labels
        ax.set_xlabel('Datamængde', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(titel, fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(x_values.astype(int))
        ax.legend(fontsize=12)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=13)

    plt.tight_layout()
    kørsel_path = kørsel_paths['norm']
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{fignavn['norm']}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{fignavn['norm']}.pdf"))
    plt.close()

for x in fignavn.values():
    if os.path.exists(rod(x)):
        shutil.rmtree(rod(x))

stjerner = viz0.get_stjerner()
print(stjerner)
groups = [f'eksp2_{stjerne}' for stjerne in stjerner]

# groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    group_df = viz0.get_group_df(group)
    fortræningsudgaver, temperaturer, seeds = viz0.get_loop_params_group_df(group_df)
    eftertræningsmængder = group_df['eftertræningsmængde'].unique()
    assert len(temperaturer) == 1
    temperatur = list(temperaturer)[0]

    kørsel_paths = {}
    for k, v in fignavn.items():
        kørsel_paths[k] = os.path.join(rod(v), group)
        os.makedirs(kørsel_paths[k])

    # runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave=None, seed=None), runs_in_group))
    idxs = group_df['temperatur'] == temperatur
    df = group_df[idxs]
    # df = viz0.get_df(runs_filtered)

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']}):
        plot(df, fortræningsudgaver)
        plot_normalisere_enkelt(df, fortræningsudgaver)
