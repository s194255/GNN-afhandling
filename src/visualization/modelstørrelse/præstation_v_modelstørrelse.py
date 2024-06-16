print("importerer")
import copy
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from src.visualization.farver import corporate_red, blue, white
# from src.visualization import viz0
from src.visualization.viz0 import get_group_df
from tqdm import tqdm
import numpy as np
# from scipy.stats import norm, ttest_ind
# import statsmodels.api as sm
# import pylab as py
# from seaborn import violinplot
print("færdig med at importere")

#%%

rod = 'reports/figures/Eksperimenter/modelstørrelse/modelstørrelse'
temperaturer = ['frossen', 'optøet']
FIGNAVN = 'modelstørrelse'
groups = ['eksp5_0']
farveopslag = {
    'frossen': blue,
    'optøet': corporate_red
}

if os.path.exists(rod):
    shutil.rmtree(rod)


def plot(group_df: pd.DataFrame):
    golden_ratio = (5 ** .5 - 1) / 2
    w = 14
    h = 6
    plt.figure(figsize=(w, h))
    plt.plot(group_df['_runtime'], group_df['test_loss_mean'],
             marker='o', color=corporate_red, linewidth=3, markersize=40)

    # Tilføj labels og title
    plt.xlabel('Kørselstid (sek)', fontsize=20)
    plt.ylabel('MAE', fontsize=20)

    for idx, row in group_df.iterrows():
        plt.text(row['_runtime'], row['test_loss_mean'],
                 f"{row['hidden_channels']:.0f}",
                 fontsize=18, ha='center', va='center', color=white)

    log_log = True
    ax = plt.gca()
    if log_log:
        x_margin = 100
        y_margin = 100
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([max(group_df['_runtime'].min() - x_margin, 1), group_df['_runtime'].max() + 100*x_margin])
        ax.set_ylim([group_df['test_loss_mean'].min() - y_margin, group_df['test_loss_mean'].max() + 50*y_margin])

        x_ticks = group_df['_runtime'].unique()
        ax.set_xticks(x_ticks)
        x_ticks = list(map(lambda x: f'{x:.2e}', x_ticks))
        ax.set_xticklabels(x_ticks, fontsize=13)

        y_ticks = np.geomspace(group_df['test_loss_mean'].min(), group_df['test_loss_mean'].max(), 10)
        ax.set_yticks(y_ticks)
        y_ticks = list(map(lambda x: f'{x:.2f}', y_ticks))
        ax.set_yticklabels(y_ticks, fontsize=13)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.tick_params(axis='x', labelrotation=-20, which='minor')
        ax.tick_params(axis='x', labelrotation=-20, which='major')
    else:
        x_margin = 1000
        y_margin = 1000
        ax.set_xlim([group_df['_runtime'].min() - x_margin, group_df['_runtime'].max() + x_margin])
        ax.set_ylim([group_df['test_loss_mean'].min() - y_margin, group_df['test_loss_mean'].max() + y_margin])

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)


    # Tilføj grid og vis plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))
    plt.close()

def plot2(group_df: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for i,col in enumerate(['hidden_channels', '_runtime']):
        axs[i].plot(group_df[col], group_df['test_loss_mean'],
                    marker='o', color=corporate_red, linewidth=3)

        axs[i].set_xlabel('Hidden Channels', fontsize=14)
        axs[i].set_ylabel('Test Loss Mean', fontsize=14)

        if i == 0:
            axs[i].set_xticks(group_df[col])
        axs[i].grid(True)


    # Tilføj grid og vis plot
    plt.grid(True)
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))


for group in tqdm(groups):
    kørsel_path = os.path.join(rod, group)
    os.makedirs(kørsel_path)

    group_df = get_group_df(group)
    plot(group_df)

