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

TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet rygrad"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9 fortræning',
            '3D-EMGP-lokalt': '3D-EMGP kun lokalt',
            '3D-EMGP-globalt': '3D-EMGP kun globalt',
            '3D-EMGP-begge': '3D-EMGP'
            }



# FIGNAVN = 'søjle'
ROOT = os.path.join('reports/figures/Eksperimenter/4')

farver = [far.corporate_red, far.blue, far.yellow, far.navy_blue, far.bright_green, far.orange]
stjerner = {
    'optøet': 'eksp2_83',
    'frossen': 'eksp2_88'
}


def plot_kernel_baseline(ax, x_values, x, farve):
    kernel = viz0.kernel_baseline()
    # x = np.linspace(30, 500, 1000)
    y = kernel(x_values)
    ax.scatter(x, y, color=farve, marker="d", label="kernel baseline", s=80,
               edgecolor=far.black)


def plot(group_df):
    # Opsætning for søjlerne
    x_values = group_df['eftertræningsmængde'].unique()
    x_values.sort()

    temperaturer = ['frossen', 'optøet']
    # fortræningsudgaver = df['fortræningsudgave'].unique()
    num_models = len(temperaturer)


    bar_width = 0.3
    x = np.arange(len(x_values))

    # Opret figuren og akserne
    fig, ax = plt.subplots(figsize=(9, 7))
    # fig, ax = plt.subplots()

    # Plot søjlerne og prikkerne
    for i in range(num_models):

        # fortræningsudgave = fortræningsudgaver[i]
        temperatur = temperaturer[i]
        målinger = group_df[group_df['temperatur'] == temperatur][['eftertræningsmængde', 'test_loss_mean']]
        søjlehøjde = målinger.groupby('eftertræningsmængde').mean().reset_index()['test_loss_mean']
        if len(søjlehøjde) != len(x_values):
            continue
        bars = ax.bar(x + (i + 0.5 - num_models / 2) * bar_width, søjlehøjde,
                      bar_width, color=farver[i], alpha=0.85)
        for j in range(len(x_values)):
            prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]]['test_loss_mean']
            n2 = len(prikker)
            label = temperatur if j==0 else None
            ax.scatter([x[j] + (i + 0.5 - num_models / 2) * bar_width] * n2, prikker,
                       color=farver[i], label=label, marker='o', edgecolor='black', alpha=1.0)

    plot_kernel_baseline(ax, x_values, x, farver[i+1])

    # Tilpasning af akserne og labels
    ax.set_xlabel('Datamængde', fontsize=16)
    ax.set_ylabel('MAE', fontsize=16)
    ax.set_title('Frossen versus optøet', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values)
    ax.set_yscale("log")
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    ax.yaxis.set_minor_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.tight_layout()

    plt.savefig(os.path.join(rod(group), f"{temperatur}_trænusik2.jpg"))
    plt.savefig(os.path.join(rod(group), f"{temperatur}_trænusik2.pdf"))
    plt.close()

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)
#
# runs = wandb.Api().runs("afhandling")
# runs = list(filter(lambda w: viz0.is_suitable(w, 'eksp2'), runs))
# df = None
# kørsel_path = os.path.join(ROOT)
# os.makedirs(kørsel_path)
#
# for temperatur in ['frossen', 'optøet']:
#     group = stjerner[temperatur]
#     runs_in_group, fortræningsudgaver, temperaturer_lp, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
#     eftertræningsmængder = viz0.get_eftertræningsmængder(group, runs)
#     assert len(temperaturer_lp) == 1
#     assert list(temperaturer_lp)[0] == temperatur
#     # kørsel_path = os.path.join(ROOT, f'{}'
#
#     runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur,
#                                                            fortræningsudgave='3D-EMGP-lokalt', seed=None), runs_in_group))
#     df_linje = viz0.get_df(runs_filtered)
#     if df is None:
#         df = {col: [] for col in df_linje.columns}
#         df = pd.DataFrame(data=df)
#     df = pd.concat([df, pd.DataFrame(data=df_linje)], ignore_index=True)
# print(df)
# plot(df)


    # plot(df_linje, fortræningsudgaver)

groups = ['eksp2_0']
kørsel_path = os.path.join(ROOT)
os.makedirs(kørsel_path)

# groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    group_root = os.path.join(ROOT, group)
    if os.path.exists(group_root):
        shutil.rmtree(group_root)
    os.makedirs(group_root, exist_ok=True)
    group_df = viz0.get_group_df(group)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']}):
        # plot(df, fortræningsudgaver)
        # plot_normalisere_enkelt(df, fortræningsudgaver)
        # trænusik4(df, fortræningsudgaver)
        # samfattabelmager(df, fortræningsudgaver)
        pass