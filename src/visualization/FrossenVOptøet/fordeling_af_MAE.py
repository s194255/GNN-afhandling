print("importerer")
import copy
import shutil
import os
import matplotlib.pyplot as plt
from src.visualization.farver import corporate_red, blue
from src.visualization import viz0
from tqdm import tqdm
import numpy as np
from scipy.stats import norm, ttest_ind
import statsmodels.api as sm
import pylab as py
from seaborn import violinplot
print("færdig med at importere")

rod = 'reports/figures/Eksperimenter/FrossenVOptøet/fordeling_af_MAE'
temperaturer = ['frossen', 'optøet']
FIGNAVN = 'fordeling_af_MAE'
groups = ['eksp4_3']
farveopslag = {
    'frossen': blue,
    'optøet': corporate_red
}
templabeller = {
    'frossen': 'Frossen',
    'optøet': 'Optøet',
}

if os.path.exists(rod):
    shutil.rmtree(rod)

plot_hist = False

def plot_violin(dfs: dict, ax: plt.Axes):
    col = 'test_loss_mean'
    temperaturer = list(dfs.keys())
    df = {templabeller[temp]: dfs[temp][col] for temp in temperaturer}
    palette = [farveopslag[temp] for temp in temperaturer]

    violinplot(data=df, palette=palette, ax=ax, inner='box')

    ax.set_xticks(range(len(df.keys())))
    ax.set_xticklabels(df.keys(), fontsize=13)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=13)
    ax.set_ylabel('MAE', fontsize=15)
    # ax.set_title('Violin', fontsize=22)

def plothist(dfs: dict, ax: plt.Axes):
    temperaturer = dfs.keys()
    for i, temperatur in enumerate(temperaturer):
        df = dfs[temperatur]
        col = 'test_loss_mean'

        mean = df[col].mean()
        std_dev = df[col].std()
        print(f"Antal datapunkter for {temperatur}: {len(df)}")
        print(f"Mean for {temperatur}: {mean}")
        print(f"Std for {temperatur}: {std_dev}")
        print("\n")

        x = np.linspace(df[col].min(), df[col].max(), 100)
        fit = norm.pdf(x, mean, std_dev)
        color = farveopslag[temperatur]

        bins = 10
        ax.hist(df[col], bins=bins, density=True, alpha=0.60, color=color, edgecolor='black',
                label=temperatur)  # Histogram

    ax.set_xlabel("MAE", fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=13)
    # ax.set_title('Histogram', fontsize=22)
    ax.legend(fontsize=16)
    ax.grid(True)
    ax.yaxis.set_ticks([])

def welsh_t_test(dfs):
    col = 'test_loss_mean'
    a = dfs['frossen'][col]
    b = dfs['optøet'][col]
    alternative = 'two-sided'
    equal_var = False
    t, p = ttest_ind(a, b, alternative=alternative, equal_var=equal_var)
    for t in ['frossen', 'optøet']:
        print(f"temperatur = {t}")
        print(f"mean = {dfs[t][col].mean()}")
        print(f"std = {dfs[t][col].std()}")
        print(f"runtime = {dfs[t]['_runtime'].mean()}")
        print("\n")
    # print(f"middelværdi af ingen fortræning = {np.mean(a)}")
    # print(f"middelværdi af 3D-EMGP-lokalt = {np.mean(b)}")
    print(f"p-værdi = {p}")

def qqplot(dfs):
    temperaturer = dfs.keys()
    for i, temperatur in enumerate(temperaturer):
        df = dfs[temperatur]
        col = 'test_loss_mean'

        sm.qqplot(df[col])
        py.savefig(os.path.join(kørsel_path, f"qq_{temperatur}.jpg"))

        sm.qqplot(np.log(df[col]))
        py.savefig(os.path.join(kørsel_path, f"qq_log_{temperatur}.jpg"))




# groups, runs = viz0.get_groups_runs('eksp4')
for group in tqdm(groups):
    group_df = viz0.get_group_df(group)
    fortræningsudgaver, temperaturer_lp, seeds = viz0.get_loop_params_group_df(group_df)

    assert len(fortræningsudgaver) == 1
    temperaturer_lp = [temperatur for temperatur in temperaturer if temperatur in temperaturer_lp]
    fortræningsudgave = list(fortræningsudgaver)[0]
    kørsel_path = os.path.join(rod, group)
    os.makedirs(kørsel_path)

    dfs = {}
    for i, temperatur in enumerate(temperaturer_lp):
        df = copy.deepcopy(group_df)
        df = df[df['temperatur'] == temperatur]
        df = df[df['fortræningsudgave'] == fortræningsudgave]
        dfs[temperatur] = df

    if plot_hist:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        axs = axs.ravel()

        plothist(dfs, axs[0])
        plot_violin(dfs, axs[1])
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        # axs = axs.ravel()

        # plothist(dfs, )
        plot_violin(dfs, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))
    welsh_t_test(dfs)
