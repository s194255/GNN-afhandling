print("importerer ...")
import shutil
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from src.visualization.farver import farver, blue, corporate_red, bright_green
from src.visualization.viz0 import get_group_df, get_loop_params_group_df, set_size
import pandas as pd
from tqdm import tqdm
import copy
import numpy as np
print("færdig med at importere ...")

def find_skærngspunkter(curves: dict, ax: plt.Axes)  -> list:
    fortræer = list(curves.keys())
    x = curves[fortræer[0]]['eftertræningsmængde']
    y1 = curves[fortræer[0]]['test_loss_mean']
    y2 = curves[fortræer[1]]['test_loss_mean']

    løsninger = []
    for i in range(1, len(x)):
        if y1[i] == y2[i]:
            løsninger.append(x[i])
        før = y1[i-1] > y2[i-1]
        efter = y1[i] > y2[i]

        if not før == efter:
            f1 = np.poly1d(np.polyfit(x[i - 1: i + 1], y1[i - 1: i + 1], 1))
            f2 = np.poly1d(np.polyfit(x[i - 1: i + 1], y2[i - 1: i + 1], 1))

            løsning = (f1.coef[1] - f2.coef[1]) / (f2.coef[0] - f1.coef[0])
            løsning = 0.98*løsning

            løsninger.append(løsning)


    return løsninger

def plot(group_df: pd.DataFrame):
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']}):
        for temperatur in temperaturer:
            golden_ratio = (5 ** .5 - 1) / 2
            width = 10
            height = width*golden_ratio
            fig, ax = plt.subplots(1, 1, figsize=(width, height))
            # fig, ax = plt.figure(figsize=(10,6))

            i = 0
            curves = {}
            for fortræningsudgave in fortræningsudgaver:
                df = copy.deepcopy(group_df)
                df = df[df['fortræningsudgave'] == fortræningsudgave]
                # df = df[df['seed'] == seed]
                df = df[df['temperatur'] == temperatur]

                df = df[['eftertræningsmængde', 'test_loss_mean']]
                df = df.groupby(by='eftertræningsmængde').mean().reset_index()

                label = LABELLER[fortræningsudgave]
                color = farveopslag[fortræningsudgave]
                ax.plot(df["eftertræningsmængde"], df[f"test_loss_mean"], label=label, color=color,
                        marker='o', zorder=2, linewidth=3)
                i += 1
                curves[fortræningsudgave] = df

            skæringspunkter = find_skærngspunkter(curves, ax)
            for skæringspunkt in skæringspunkter:
                ax.axvline(skæringspunkt, color=blue, linestyle='--', zorder=1, linewidth=3)
            ax.set_xlabel("Datamængde", fontsize=22)
            ax.set_ylabel("MAE", fontsize=22)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.yaxis.set_minor_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.legend(fontsize=18)
            # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.tick_params(axis='x', labelrotation=45, which='minor')
            ax.tick_params(axis='x', labelrotation=45, which='major')
            ticks = list(group_df['eftertræningsmængde'].unique())
            ax.set_xticks(ticks)
            # ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=13)
            ax.grid()

            plt.tight_layout()
            for ext in ['jpg', 'pdf']:
                plt.savefig(os.path.join(kørsel_path, f"{temperatur}_konvergens.{ext}"))
            plt.close()

ROOT = 'reports/figures/Eksperimenter/6'

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9-fortræning',
            '3D-EMGP-lokalt': 'Lokalt',
            '3D-EMGP-globalt': 'Globalt',
            '3D-EMGP-begge': 'Begge'
            }
farveopslag = {
    'uden': corporate_red,
    '3D-EMGP-lokalt': bright_green
}

GROUPS = ['eksp6_1']

for group in tqdm(GROUPS):
    group_df = get_group_df(group)
    fortræningsudgaver, temperaturer, seeds = get_loop_params_group_df(group_df)

    # runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    # seeds = random.sample(list(seeds), k=4)
    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)
    plot(group_df)
    # lol()