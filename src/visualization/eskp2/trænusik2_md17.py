import copy
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
import scipy.stats as st

TITLER = {'frossen': "Sammenligning (frossen)",
          'optøet': "Sammenligning"}

YLABEL = r'MAE $ma_0^3$'
XLABEL = r'Datamængde ($N_{træn}$)'
fignavn = {
    'trad': 'trænusik2',
    'norm': 'trænusik2_normaliseret'
}
rod = lambda x: os.path.join('reports/figures/Eksperimenter/2', x)
KERNELBASELINEFARVE = far.black

def plot_kernel_baseline(ax, x_values, x, farve, predicted_attribute):
    if predicted_attribute == 'MD17':
        return
    kernel = viz0.kernel_baseline(predicted_attribute)
    # x = np.linspace(30, 500, 1000)
    y = kernel(x_values)
    idxs = x_values >= 100
    ax.scatter(x[idxs], y[idxs], color=farve, marker="d", label="kernel baseline", s=80,
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

    for i in range(num_models):

        fortræningsudgave = fortræningsudgaver[i]
        målinger = df[df['fortræningsudgave'] == fortræningsudgave][['eftertræningsmængde', 'test_loss_mean']]
        søjlehøjde = målinger.groupby('eftertræningsmængde').mean().reset_index()['test_loss_mean']
        if len(søjlehøjde) != len(x_values):
            continue
        farve = viz0.FARVEOPSLAG[fortræningsudgave]
        bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, søjlehøjde,
                      bar_width, color=farve, alpha=0.85)
        for j in range(len(x_values)):
            prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]]['test_loss_mean']
            n2 = len(prikker)
            label = viz0.FORT_LABELLER[fortræningsudgave] if j==0 else None
            ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker,
                       color=farve, label=label, marker='o', edgecolor='black', alpha=1.0)

    plot_kernel_baseline(ax, x_values, x, KERNELBASELINEFARVE, predicted_attribute)

    ax.set_xlabel(XLABEL, fontsize=16)
    ax.set_ylabel(YLABEL, fontsize=16)
    ax.set_title(TITLER[temperatur], fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values.astype(int))
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    plt.tight_layout()

    plt.savefig(os.path.join(rod(group), f"{temperatur}_{fignavn['trad']}.jpg"))
    plt.savefig(os.path.join(rod(group), f"{temperatur}_{fignavn['trad']}.pdf"))
    plt.close()

def plot_normalisere_enkelt(df1, fortræningsudgaver):
    # Opsætning for søjlerne
    x_values = df1['eftertræningsmængde'].unique()
    x_values.sort()
    # fortræningsudgaver = list(set(fortræningsudgaver) - {'uden'})
    fortræningsudgaver =  copy.deepcopy(fortræningsudgaver)
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
            farve = viz0.FARVEOPSLAG[fortræningsudgave]
            bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, søjlehøjde, bar_width, color=farve,
                          alpha=0.85, zorder=2)
            for j in range(len(x_values)):
                prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]][col]
                n2 = len(prikker)
                label = viz0.FORT_LABELLER[fortræningsudgave] if j == 0 else None
                ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker, color=farve, label=label,
                           marker='o', edgecolor='black', alpha=1.0, zorder=3)
        titel = f'%-vis reduktion ift. Ingen'
        ylabel = '%'

        # Tilpasning af akserne og labels
        ax.set_xlabel(XLABEL, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(titel, fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(x_values.astype(int))
        ax.legend(fontsize=12)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(rod(group), f"{temperatur}_{fignavn['norm']}.jpg"))
    plt.savefig(os.path.join(rod(group), f"{temperatur}_{fignavn['norm']}.pdf"))
    plt.close()


def trænusik4(df, fortræer):
    # Opsætning for søjlerne
    x_values = df['eftertræningsmængde'].unique()
    x_values.sort()
    num_models = len(fortræer)
    predicted_attribute = df['predicted_attribute'].unique()
    assert len(predicted_attribute) == 1
    predicted_attribute = predicted_attribute[0]

    gray = '#DADADA'

    bar_width = 0.15
    x = np.arange(len(x_values))

    fig, ax = plt.subplots(figsize=(9, 7))

    for i in range(num_models):

        fortræ = fortræer[i]
        målinger = df[df['fortræningsudgave'] == fortræ][['eftertræningsmængde', 'test_loss_mean']]
        means = målinger.groupby('eftertræningsmængde').mean().reset_index()['test_loss_mean']
        if len(means) != len(x_values):
            continue
        farve = viz0.FARVEOPSLAG[fortræ]
        label = viz0.FORT_LABELLER[fortræ]
        bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, means,
                      bar_width, color=farve, alpha=1.0, label=label)
        conf_intervals = []
        for j in range(len(x_values)):
            prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]]['test_loss_mean']
            n2 = len(prikker)
            label = viz0.FORT_LABELLER[fortræ] if j==0 else None
            ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker,
                       color=gray, marker='.', alpha=1.0,
                       # s=20,
                       # edgecolor=farve
                       )
            conf_interval = st.norm.interval(confidence=0.90, loc=np.mean(prikker), scale=st.sem(prikker))
            conf_intervals.append(conf_interval)

        conf_intervals = np.array(conf_intervals)
        lower_errors = means - conf_intervals[:, 0]
        upper_errors = conf_intervals[:, 1] - means
        error_bars = [lower_errors, upper_errors]

        ax.errorbar(x + (i + 0.5 - num_models / 2) * bar_width, means, yerr=error_bars, fmt='none',
                    ecolor='black', elinewidth=1.5, capsize=5,
                    capthick=1.5, zorder=2)

    plot_kernel_baseline(ax, x_values, x, KERNELBASELINEFARVE, predicted_attribute)

    ax.set_xlabel(XLABEL, fontsize=16)
    ax.set_ylabel(YLABEL, fontsize=16)
    ax.set_title(TITLER[temperatur], fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values.astype(int))
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    plt.tight_layout()

    plt.savefig(os.path.join(rod(group), f"{temperatur}_trænusik2.jpg"))
    plt.savefig(os.path.join(rod(group), f"{temperatur}_trænusik2.pdf"))
    plt.close()


def samfattabelmager(df, fortræer):
    x_values = df['eftertræningsmængde'].unique()
    x_values.sort()
    inder = lambda x_value: f'{int(x_value)}'

    samfattabel = {'fortræningsudgave': []}
    samfattabel = {**samfattabel, **{inder(x_value): [] for x_value in x_values}}
    samfattabel = pd.DataFrame(data=samfattabel)
    for fortræ in fortræer:
        række = {'fortræningsudgave': [viz0.FORT_LABELLER[fortræ]]}
        for x_value in x_values:
            idxs = df['fortræningsudgave'] == fortræ
            idxs = (idxs) & (df['eftertræningsmængde'] == x_value)
            mean = df[idxs][['test_loss_mean']].mean()
            std = df[idxs][['test_loss_mean']].std()
            række[inder(x_value)] = mean
        samfattabel = pd.concat([samfattabel, pd.DataFrame(data=række)], ignore_index=True)
    latex_table = samfattabel.to_latex(index=False, float_format="%.2f")
    start = latex_table.find(viz0.FORT_LABELLER['uden'])
    end = latex_table.find(r'\end{tabular}')
    latex_table = latex_table[start:end]
    a = 2
    path = os.path.join(rod(group), f"{temperatur}_trænusik.tex")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(latex_table)



groups = ['eksp2-md17_0']

# groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    if os.path.exists(rod(group)):
        shutil.rmtree(rod(group))
    os.makedirs(rod(group), exist_ok=True)
    group_df = viz0.get_group_df(group)
    fortræningsudgaver, temperaturer, seeds = viz0.get_loop_params_group_df(group_df)
    eftertræningsmængder = group_df['eftertræningsmængde'].unique()
    assert len(temperaturer) == 1
    temperatur = list(temperaturer)[0]

    idxs = group_df['temperatur'] == temperatur
    df = group_df[idxs]

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']}):
        # plot(df, fortræningsudgaver)
        plot_normalisere_enkelt(df, fortræningsudgaver)
        trænusik4(df, fortræningsudgaver)
        samfattabelmager(df, fortræningsudgaver)