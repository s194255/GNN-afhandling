print("importerer")
import copy
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.farver import corporate_red, blue, white, bright_green
# from src.visualization import viz0
from src.visualization.viz0 import get_group_df
from tqdm import tqdm
# import numpy as np
# from scipy.stats import norm, ttest_ind
# import statsmodels.api as sm
# import pylab as py
# from seaborn import violinplot
print("færdig med at importere")


rod = 'reports/figures/Eksperimenter/7'
temperaturer = ['frossen', 'optøet']
FIGNAVN = 'effekt_v_n_lag'
groups = ['eksp7_0']
farveopslag = {
    '3D-EMGP-lokalt': bright_green,
    'uden': corporate_red
}

if os.path.exists(rod):
    shutil.rmtree(rod)


def plot(group_df: pd.DataFrame):
    # fortræer = group_df['fortræningsudgave'].unique()
    # width = 14
    # golden_ratio = (5 ** .5 - 1) / 2
    # fig, ax = plt.subplots(1, 1, figsize=(width, width*golden_ratio))
    # for fortræ in fortræer:
    #     idxs = group_df['fortræningsudgave'] == fortræ
    #     df = group_df[idxs]
    #     df = df[['num_layers', 'test_loss_mean']].groupby(by='num_layers').mean().reset_index()
    #     color = farveopslag[fortræ]
    #     ax.plot(df['num_layers'], df['test_loss_mean'], label=fortræ, marker='o',
    #             linewidth=3, markersize=10, color=color)
    #
    # ax.legend(fontsize=18)
    #
    # ax.tick_params(axis='both', which='major', labelsize=15)
    # ax.tick_params(axis='both', which='minor', labelsize=15)
    #
    # # Tilføj grid og vis plot
    # plt.grid(True)
    # plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    # plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))
    fortræer = group_df['fortræningsudgave'].unique()
    width = 14
    golden_ratio = (5 ** .5 - 1) / 2
    fig, ax = plt.subplots(1, 1, figsize=(width, width * golden_ratio))

    all_layers = set()

    for fortræ in fortræer:
        idxs = group_df['fortræningsudgave'] == fortræ
        df = copy.deepcopy(group_df[idxs])
        df = df[['num_layers', 'test_loss_mean']].groupby(by='num_layers').mean().reset_index()
        if fortræ == '3D-EMGP-lokalt':
            a = 2
        color = farveopslag[fortræ]
        ax.plot(df['num_layers'], df['test_loss_mean'], label=fortræ, marker='o',
                linewidth=3, markersize=10, color=color)
        all_layers.update(df['num_layers'].tolist())

    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    # Indstil x-ticks til kun at vise de unikke værdier af 'num_layers'
    ax.set_xticks(sorted(all_layers))

    # Tilføj grid og vis plot
    plt.grid(True)
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))



for group in tqdm(groups):
    kørsel_path = os.path.join(rod, group)
    os.makedirs(kørsel_path)

    group_df = get_group_df(group)
    plot(group_df)

