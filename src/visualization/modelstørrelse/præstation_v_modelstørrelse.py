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
from matplotlib.lines import Line2D
# from scipy.stats import norm, ttest_ind
# import statsmodels.api as sm
# import pylab as py
# from seaborn import violinplot
print("færdig med at importere")

#%%

rod = 'reports/figures/Eksperimenter/modelstørrelse/modelstørrelse'
temperaturer = ['frossen', 'optøet']
FIGNAVN = 'modelstørrelse'
groups = ['eksp5_1']
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
    plt.figure(figsize=(w, w*golden_ratio))
    group_df = group_df[['hidden_channels', 'test_loss_mean', '_runtime']]
    group_df = group_df.groupby('hidden_channels').mean().reset_index()
    plt.plot(group_df['_runtime'], group_df['test_loss_mean'],
             marker='o', color=corporate_red, linewidth=3, markersize=40)

    # Tilføj labels og title
    plt.xlabel('Kørselstid (sek)', fontsize=20)
    plt.ylabel('MAE', fontsize=20)

    for idx, row in group_df.iterrows():
        plt.text(row['_runtime'], row['test_loss_mean'],
                 f"{row['hidden_channels']:.0f}",
                 fontsize=18, ha='center', va='center', color=white)

    log_log = False
    ax = plt.gca()
    if log_log:
        ax.set_xscale("log")
        ax.set_yscale("log")

        x_margin = 0.3
        y_margin = 0.3
        ax.set_xlim([np.exp(np.log(group_df['_runtime'].min()) - x_margin),
                     np.exp(np.log(group_df['_runtime'].max()) + x_margin)])

        ax.set_ylim([np.exp(np.log(group_df['test_loss_mean'].min()) - y_margin),
                     np.exp(np.log(group_df['test_loss_mean'].max()) + y_margin)])

        # ax.set_ylim([group_df['test_loss_mean'].min() - y_margin, group_df['test_loss_mean'].max() + 50*y_margin])

        x_ticks = group_df['_runtime'].unique()
        ax.set_xticks(x_ticks)
        x_ticks = list(map(lambda x: f'{x:.0f}', x_ticks))
        ax.set_xticklabels(x_ticks, fontsize=13)

        y_ticks = np.geomspace(group_df['test_loss_mean'].min(), group_df['test_loss_mean'].max(), 10)
        ax.set_yticks(y_ticks)
        y_ticks = list(map(lambda x: f'{x:.0f}', y_ticks))
        ax.set_yticklabels(y_ticks, fontsize=13)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.tick_params(axis='x', labelrotation=-20, which='minor')
        ax.tick_params(axis='x', labelrotation=-20, which='major')
        ax.minorticks_off()
    else:
        x_margin = 100
        y_margin = 100
        ax.set_xlim([group_df['_runtime'].min() - x_margin, group_df['_runtime'].max() + x_margin])
        ax.set_ylim([group_df['test_loss_mean'].min() - y_margin, group_df['test_loss_mean'].max() + y_margin])

        y_ticks = np.linspace(group_df['test_loss_mean'].min(), group_df['test_loss_mean'].max(), 15)
        ax.set_yticks(y_ticks)

        x_ticks = np.linspace(group_df['_runtime'].min(), group_df['_runtime'].max(), 20)
        ax.set_xticks(x_ticks)

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        ax.tick_params(axis='x', labelrotation=-20, which='minor')
        ax.tick_params(axis='x', labelrotation=-20, which='major')


    # Tilføj grid og vis plot
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.svg"))
    plt.close()

def tabel(group_df: pd.DataFrame):
    hcs = sorted(group_df['hidden_channels'].unique())
    for hc in hcs:
        idxs = group_df['hidden_channels'] == hc
        mean = group_df[idxs]['test_loss_mean'].mean()
        std = group_df[idxs]['test_loss_mean'].std()
        runtime = group_df[idxs]['_runtime'].mean()
        n = len(group_df[idxs]['test_loss_mean'])
        print(f"hidden_channels = {hc}")
        print(f"mean = {mean}")
        print(f"std = {std}")
        print(f"runtime = {runtime}")
        print(f"n = {n}")
        print("\n")


for group in tqdm(groups):
    kørsel_path = os.path.join(rod, group)
    os.makedirs(kørsel_path)

    group_df = get_group_df(group)
    plot(group_df)
    tabel(group_df)

