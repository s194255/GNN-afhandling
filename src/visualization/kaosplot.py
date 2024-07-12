import copy
import shutil
import os
import matplotlib.pyplot as plt
# from src.visualization.farver import farver
from matplotlib.ticker import ScalarFormatter, MultipleLocator, AutoMinorLocator
import src.visualization.farver as far
from tqdm import tqdm
from src.visualization import viz0
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

def sejrsstatistik(group_df: pd.DataFrame):
    # # Filtrering af NaN og inf værdier
    # filtered_df = group_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['test_loss_mean'])
    group_df = group_df.replace([np.inf, -np.inf], np.nan)

    # Initialisering af tællere
    ingen_fortraening_bedst = 0
    lokal_global_bedst = 0
    testmaaling_ended_i_nan = 0

    # Unikke seeds
    seeds = group_df['seed'].unique()

    for seed in seeds:
        seed_df = group_df[group_df['seed'] == seed]

        if seed_df['test_loss_mean'].isna().any():
            testmaaling_ended_i_nan += 1
            continue

        min_row = seed_df.loc[seed_df['test_loss_mean'].idxmin()]
        winner = min_row['fortræningsudgave']

        if winner == "uden":
            ingen_fortraening_bedst += 1
        elif winner == "3D-EMGP-begge":
            lokal_global_bedst += 1

    # Beregning af frekvenser
    total_seeds = len(seeds)
    ingen_fortraening_frekvens = ingen_fortraening_bedst / total_seeds
    lokal_global_frekvens = lokal_global_bedst / total_seeds
    testmaaling_nan_frekvens = testmaaling_ended_i_nan / total_seeds

    # Udskrivning af værdier
    print(f"Ingen fortræning bedst: {ingen_fortraening_bedst}, Frekvens: {ingen_fortraening_frekvens:.2f}")
    print(f"Lokal+Global bedst: {lokal_global_bedst}, Frekvens: {lokal_global_frekvens:.2f}")
    print(f"Testmåling endte i NAN: {testmaaling_ended_i_nan}, Frekvens: {testmaaling_nan_frekvens:.2f}")

def nan_statistik(group_df: pd.DataFrame):
    # Erstat inf med NaN for ensartet håndtering
    group_df = group_df.replace([np.inf, -np.inf], np.nan)

    # Filtrer data for 'Ingen fortræning' og '3d-emgp-begge'
    ingen_fortræning_df = group_df[group_df['fortræningsudgave'] == 'uden']
    emgp_begge_df = group_df[group_df['fortræningsudgave'] == '3D-EMGP-begge']

    # Find unikke seeds
    seeds = group_df['seed'].unique()

    # Initialisere tællere
    count_ingen_nan_emgp_not_nan = 0
    count_emgp_nan_ingen_not_nan = 0
    count_both_nan = 0

    # Gå igennem hver seed og udfør tællinger
    for seed in seeds:
        ingen_nan = ingen_fortræning_df[
                        (ingen_fortræning_df['seed'] == seed) & (ingen_fortræning_df['test_loss_mean'].isna())].shape[
                        0] > 0
        emgp_nan = emgp_begge_df[(emgp_begge_df['seed'] == seed) & (emgp_begge_df['test_loss_mean'].isna())].shape[
                       0] > 0

        if ingen_nan and not emgp_nan:
            count_ingen_nan_emgp_not_nan += 1
        elif emgp_nan and not ingen_nan:
            count_emgp_nan_ingen_not_nan += 1
        elif ingen_nan and emgp_nan:
            count_both_nan += 1

    # Oprette DataFrame med resultaterne
    statistik_df = pd.DataFrame({
        'antal seeds - Ingen fortræning NaN/inf, 3d-emgp-begge ikke NaN/inf': [count_ingen_nan_emgp_not_nan],
        'antal seeds - 3d-emgp-begge NaN/inf, Ingen fortræning ikke NaN/inf': [count_emgp_nan_ingen_not_nan],
        'antal seeds - Både Ingen fortræning og 3d-emgp-begge NaN/inf': [count_both_nan]
    })
    statistik_df.to_csv(os.path.join(gruppe_rod, "nanstat.csv"), index=False)
    # return statistik_df

def plot_kaos(group_df: pd.DataFrame):
    group_df = group_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['test_loss_mean'])
    fortræer = group_df['fortræningsudgave'].unique()
    seeds = group_df['seed'].unique()
    # winners = {fortræ: [] for fortræ in fortræer}
    winners = pd.DataFrame({'Vindere': [], 'MAE': []})
    for seed in seeds:
        idxs = group_df['seed'] == seed
        filtered_df = group_df[idxs]

        if filtered_df.empty:
            continue

        winner_row = filtered_df.loc[filtered_df['test_loss_mean'].idxmin()]
        winner = winner_row['fortræningsudgave']
        point = winner_row['test_loss_mean']
        # winners[winner].append(point)
        række = {'Vindere': [viz0.FORT_LABELLER[winner]], 'MAE': [point]}
        winners = pd.concat([winners, pd.DataFrame(række)])

    # winners = winners.replace([np.inf, -np.inf], np.nan).dropna(subset=['test_loss_mean'])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    # sns.set(font_scale=1)
    palette = {viz0.FORT_LABELLER[fortræ]: viz0.FARVEOPSLAG[fortræ] for fortræ in fortræer}
    sns.stripplot(data=winners, x='MAE', jitter=True, size=15, ax=ax, log_scale=True, dodge=True,
                  hue='Vindere', palette=palette, edgecolor='black', linewidth=1, legend='full')

    x_ticks = np.geomspace(winners['MAE'].min(), winners['MAE'].max(), num=10)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()


    plt.tight_layout()
    plt.savefig(os.path.join(gruppe_rod, "vindere.jpg"))
    plt.savefig(os.path.join(gruppe_rod, "vindere.pdf"))
    plt.close()



GROUPS = ['kaos_0']
ROOT = 'reports/figures/Eksperimenter/kaos'

for group in GROUPS:
    gruppe_rod = os.path.join(ROOT, group)
    os.makedirs(gruppe_rod, exist_ok=True)
    group_df = viz0.get_group_df(group, remove_nan=False)
    plot_kaos(group_df)
    nan_statistik(group_df)
    sejrsstatistik(group_df)
