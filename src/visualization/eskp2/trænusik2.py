import shutil
import os
import matplotlib.pyplot as plt
# from src.visualization.farver import farver
from matplotlib.ticker import ScalarFormatter
import src.visualization.farver as far
from tqdm import tqdm
import viz0
import numpy as np

TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet rygrad"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9 fortræning',
            '3D-EMGP-lokalt': '3D-EMGP kun lokalt',
            '3D-EMGP-globalt': '3D-EMGP kun globalt',
            '3D-EMGP-begge': '3D-EMGP'
            }

FIGNAVN = 'trænusik2'
ROOT = os.path.join('reports/figures/Eksperimenter/2', FIGNAVN)

farver = [far.corporate_red, far.blue, far.navy_blue, far.bright_green, far.orange, far.yellow]
stjerner = viz0.get_stjerner()


def plot_kernel_baseline(ax, x_values, x, farve):
    kernel = viz0.kernel_baseline()
    # x = np.linspace(30, 500, 1000)
    y = kernel(x_values)
    ax.scatter(x, y, color=farve, marker="d", label="kernel baseline", s=80,
               edgecolor=far.black)


def plot(df, fortræningsudgaver):
    # Opsætning for søjlerne
    x_values = df['eftertræningsmængde'].unique()
    x_values.sort()
    # fortræningsudgaver = df['fortræningsudgave'].unique()
    num_models = len(fortræningsudgaver)


    bar_width = 0.15
    x = np.arange(len(x_values))

    # Opret figuren og akserne
    fig, ax = plt.subplots(figsize=(9, 7))
    # fig, ax = plt.subplots()

    # Plot søjlerne og prikkerne
    for i in range(num_models):

        fortræningsudgave = fortræningsudgaver[i]
        print(fortræningsudgave)
        målinger = df[df['fortræningsudgave'] == fortræningsudgave][['eftertræningsmængde', 'test_loss_mean']]
        søjlehøjde = målinger.groupby('eftertræningsmængde').mean().reset_index()['test_loss_mean']
        if len(søjlehøjde) != len(x_values):
            continue
        # bars = ax.bar(x + (i+0.5 - num_models / 2) * bar_width, søjlehøjde,
        #               bar_width, edgecolor=farver[i], alpha=0.999, facecolor='white', linewidth=4)
        bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, søjlehøjde,
                      bar_width, color=farver[i], alpha=0.80)
        for j in range(len(x_values)):
            prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]]['test_loss_mean']
            n2 = len(prikker)
            label = LABELLER[fortræningsudgave] if j==0 else None
            # ax.scatter([x[j] + (i+0.5 - num_models / 2) * bar_width] * n2, prikker,
            #            color=farver[i], label=label, marker='x', alpha=0.5, edgecolor=far.black)
            ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker,
                       color=farver[i], label=label, marker='o', edgecolor='black', alpha=1.0)

    plot_kernel_baseline(ax, x_values, x, farver[i+1])

    # Tilpasning af akserne og labels
    ax.set_xlabel('Datamængde', fontsize=16)
    ax.set_ylabel('MAE', fontsize=16)
    ax.set_title(f'{temperatur} rygrad', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values)
    ax.set_yscale("log")
    ax.legend(fontsize=12)
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.tight_layout()

    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.pdf"))

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    if stjerner != None:
        if group not in [f'eksp2_{udvalgt}' for udvalgt in stjerner]:
            continue
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    print(fortræningsudgaver)
    eftertræningsmængder = viz0.get_eftertræningsmængder(group, runs)
    assert len(temperaturer) == 1
    temperatur = list(temperaturer)[0]
    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)

    runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave=None, seed=None), runs_in_group))
    df = viz0.get_df(runs_filtered)

    plot(df, fortræningsudgaver)
