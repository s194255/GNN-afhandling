print("importerer")
import copy
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from src.visualization.farver import corporate_red, blue, white
# from src.visualization import viz0
from src.visualization.viz0 import get_group_df
from tqdm import tqdm
# import numpy as np
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
    plt.figure(figsize=(14, 6))
    plt.plot(group_df['_runtime'], group_df['test_loss_mean'],
             marker='o', color=corporate_red, linewidth=3, markersize=40)

    # Tilføj labels og title
    plt.xlabel('Kørtid (sek)', fontsize=20)
    plt.ylabel('MAE', fontsize=20)

    for idx, row in group_df.iterrows():
        plt.text(row['_runtime'], row['test_loss_mean'],
                 f"{row['hidden_channels']:.0f}",
                 fontsize=18, ha='center', va='center', color=white)

    x_margin = 1000
    y_margin = 1000
    plt.gca().set_xlim([group_df['_runtime'].min() - x_margin, group_df['_runtime'].max() + x_margin])
    plt.gca().set_ylim([group_df['test_loss_mean'].min() - y_margin, group_df['test_loss_mean'].max() + y_margin])

    plt.gca().tick_params(axis='both', which='major', labelsize=15)
    plt.gca().tick_params(axis='both', which='minor', labelsize=15)

    # Tilføj grid og vis plot
    plt.grid(True)
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))

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

