import copy
import shutil
import os
import matplotlib.pyplot as plt
import src.visualization.farver as far
from tqdm import tqdm
from src.visualization import viz0
import numpy as np
import pandas as pd
import scipy.stats as st

TITLER = {'frossen': "Sammenligning (frossen)",
          'optøet': "Sammenligning"}

YLABEL = r'MAE'
XLABEL = r'Datamængde ($N_{træn}$)'
fignavn = {
    'trad': 'trænusik2',
    'norm': 'trænusik2_normaliseret'
}
cols_to_titles = {
    'test_energy_loss': 'Energi',
    'test_force_loss': 'Kræfter',
}

rod = lambda x: os.path.join('reports/figures/Eksperimenter/2', x)
KERNELBASELINEFARVE = far.yellow

def plot_kernel_baseline(ax, x_values, x, farve, predicted_attribute):
    kernel = viz0.kernel_baseline(predicted_attribute)
    # x = np.linspace(30, 500, 1000)
    y = kernel(x_values)
    idxs = x_values >= 0
    ax.scatter(x[idxs], y[idxs], color=farve, marker="d", label="kernel baseline", s=120,
               edgecolor=far.black)

def plot_normaliseret(df1, fortræningsudgaver, cols):
    # Opsætning for søjlerne
    x_values = df1['eftertræningsmængde'].unique()
    x_values.sort()
    # fortræningsudgaver = list(set(fortræningsudgaver) - {'uden'})
    fortræningsudgaver = copy.deepcopy(fortræningsudgaver)
    fortræningsudgaver.remove('uden')
    num_models = len(fortræningsudgaver)

    for col in cols:
        predicted_attribute = col.split("_")[1]

        df2 = df1[['eftertræningsmængde', col]][df1['fortræningsudgave'] == 'uden']
        df2 = df2.groupby(by='eftertræningsmængde').mean().reset_index()

        df1 = pd.merge(df1, df2, on='eftertræningsmængde', suffixes=('', '_df2'))

        # Normaliser 'test_loss' i df1 med 'test_loss' fra df2
        # df1['normalized_test_loss'] = 100 * df1['test_loss_mean'] / df1['test_loss_mean_df2']
        norm_col = f'normalized_{col}'
        df1[norm_col] = 100 * (df1[f'{col}_df2'] - df1[col]) / df1[f'{col}_df2']

        bar_width = 0.15
        x = np.arange(len(x_values))

        # Opret figuren og aksern
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))

        # background
        background = viz0.predicted_attribute_to_background[predicted_attribute]
        for key in ['top', 'bottom', 'left', 'right']:
            ax.spines[key].set_color(background)
            ax.spines[key].set_linewidth(4)

        # Plot søjlerne og prikkerne
        for i in range(num_models):
            fortræningsudgave = fortræningsudgaver[i]
            målinger = df1[df1['fortræningsudgave'] == fortræningsudgave][['eftertræningsmængde', norm_col]]
            søjlehøjde = målinger.groupby('eftertræningsmængde').mean().reset_index()[norm_col]

            if len(søjlehøjde) != len(x_values):
                continue
            farve = viz0.FARVEOPSLAG[fortræningsudgave]
            bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, søjlehøjde, bar_width, color=farve,
                          alpha=0.85, zorder=2)
            for j in range(len(x_values)):
                prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]][norm_col]
                n2 = len(prikker)
                label = viz0.FORT_LABELLER[fortræningsudgave] if j == 0 else None
                ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker, color=farve, label=label,
                           marker='o', edgecolor='black', alpha=1.0, zorder=3)
        # titel = f'{cols_to_titles[col]}: %-vis forbedring'
        titel = f'{cols_to_titles[col]}: %-vis forbedring ift. ' + r'$\it{Ingen\ fortræning}$'
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
        plt.savefig(os.path.join(rod(group), 'jpger', f"{cols_to_titles[col]}_{fignavn['norm']}.jpg"))
        plt.savefig(os.path.join(rod(group), f"{cols_to_titles[col]}_{fignavn['norm']}.pdf"))
        plt.close()


def trænusik4(df, fortræer, cols):
    # Opsætning for søjlerne
    df = copy.deepcopy(df)
    df = viz0.remove_nan(df, cols)
    df.loc[df['predicted_attribute'] == 'force', 'predicted_attribute'] = 'MD17'
    x_values = df['eftertræningsmængde'].unique()
    x_values.sort()
    num_models = len(fortræer)
    farveopslag = copy.deepcopy(viz0.FARVEOPSLAG)
    farveopslag['3D-EMGP-begge'] = '#1c2761'

    gray = '#DADADA'

    bar_width = 0.15
    x = np.arange(len(x_values))

    for col in cols:
        predicted_attribute = col.split("_")[1]

        fig, ax = plt.subplots(figsize=(9, 7))

        # background
        background = viz0.predicted_attribute_to_background[predicted_attribute]
        # ax.set_facecolor(background)
        for key in ['top', 'bottom', 'left', 'right']:
            ax.spines[key].set_color(background)
            ax.spines[key].set_linewidth(4)


        for i in range(num_models):

            fortræ = fortræer[i]
            målinger = df[df['fortræningsudgave'] == fortræ][['eftertræningsmængde', col]]
            means = målinger.groupby('eftertræningsmængde').mean().reset_index()[col]
            if len(means) != len(x_values):
                continue
            farve = farveopslag[fortræ]
            label = viz0.FORT_LABELLER[fortræ]
            bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, means,
                          bar_width, color=farve, alpha=1.0, label=label)
            conf_intervals = []
            for j in range(len(x_values)):
                prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]][col]
                n2 = len(prikker)
                label = viz0.FORT_LABELLER[fortræ] if j==0 else None
                ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker,
                           color=gray, marker='.', alpha=1.0,
                           s=150,
                           edgecolor=far.black
                           )
                conf_interval = st.norm.interval(confidence=0.90, loc=np.mean(prikker), scale=st.sem(prikker))
                conf_intervals.append(conf_interval)

            conf_intervals = np.array(conf_intervals)
            lower_errors = means - conf_intervals[:, 0]
            upper_errors = conf_intervals[:, 1] - means
            error_bars = [lower_errors, upper_errors]

            ax.errorbar(x + (i + 0.5 - num_models / 2) * bar_width, means, yerr=error_bars, fmt='none',
                        ecolor='black', elinewidth=3.0, capsize=10,
                        capthick=2.0, zorder=2)

        plot_kernel_baseline(ax, x_values, x, KERNELBASELINEFARVE, predicted_attribute)

        ax.grid(alpha=0.6)
        ax.set_xlabel(XLABEL, fontsize=16)
        ax.set_ylabel(YLABEL, fontsize=16)
        titel = cols_to_titles[col]
        ax.set_title(titel, fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(x_values.astype(int))
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=13)
        plt.tight_layout()

        plt.savefig(os.path.join(rod(group), 'jpger', f"{titel}_trænusik2.jpg"))
        plt.savefig(os.path.join(rod(group), f"{titel}_trænusik2.pdf"))
        plt.close()




def samfattabelmager(df, fortræer, cols):
    x_values = df['eftertræningsmængde'].unique()
    x_values.sort()
    inder = lambda x_value: f'{int(x_value)}'

    for col in cols:
        predicted_attribute = predicted_attribute = col.split("_")[1]

        samfattabel = {'fortræningsudgave': []}
        samfattabel = {**samfattabel, **{inder(x_value): [] for x_value in x_values}}
        samfattabel = pd.DataFrame(data=samfattabel)
        for fortræ in fortræer:
            række = {'fortræningsudgave': [viz0.FORT_LABELLER[fortræ]]}
            for x_value in x_values:
                idxs = df['fortræningsudgave'] == fortræ
                idxs = (idxs) & (df['eftertræningsmængde'] == x_value)
                mean = df[idxs][[col]].mean()
                std = df[idxs][[col]].std()
                række[inder(x_value)] = mean
            samfattabel = pd.concat([samfattabel, pd.DataFrame(data=række)], ignore_index=True)

        kernel = viz0.kernel_baseline(predicted_attribute)
        y = kernel(x_values)
        række = {'fortræningsudgave': ['kernel baseline']}
        række = {**række, **{inder(x_values[i]): [y[i]] for i in range(len(x_values))}}
        samfattabel = pd.concat([samfattabel, pd.DataFrame(data=række)], ignore_index=True)


        latex_table = samfattabel.to_latex(index=False, float_format="%.3f")
        start = latex_table.find(viz0.FORT_LABELLER['uden'])
        end = latex_table.find(r'\end{tabular}')
        latex_table = latex_table[start:end]
        path = os.path.join(rod(group), f"{cols_to_titles[col]}_trænusik.tex")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(latex_table)



groups = ['eksp2-md17_2', 'eksp2-force_0']

# groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    if os.path.exists(rod(group)):
        shutil.rmtree(rod(group))
    os.makedirs(rod(group), exist_ok=True)
    os.makedirs(os.path.join(rod(group), 'jpger'))
    group_df = viz0.get_group_df(group, remove_nan=False)
    fortræningsudgaver, temperaturer, seeds = viz0.get_loop_params_group_df(group_df)
    eftertræningsmængder = group_df['eftertræningsmængde'].unique()
    assert len(temperaturer) == 1
    temperatur = list(temperaturer)[0]

    idxs = group_df['temperatur'] == temperatur
    df = group_df[idxs]

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']}):
        # plot_normalisere_enkelt(df, fortræningsudgaver)
        # trænusik4(df, fortræningsudgaver, 'test_energy_loss')
        trænusik4(df, fortræningsudgaver, ['test_force_loss', 'test_energy_loss'])
        samfattabelmager(df, fortræningsudgaver, ['test_force_loss', 'test_energy_loss'])
        plot_normaliseret(df, fortræningsudgaver, ['test_force_loss', 'test_energy_loss'])