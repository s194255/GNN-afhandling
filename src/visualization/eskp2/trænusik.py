import shutil
import os
import matplotlib.pyplot as plt
from src.visualization.farver import farver
import pandas as pd
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
from src.visualization import viz0

TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet rygrad"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'snydefortræning'}

FIGNAVN = 'trænusik'
ROOT = os.path.join('reports/figures/Eksperimenter/2', FIGNAVN)



if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    # if group not in ['eksp2_68']:
    #     continue
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    print(fortræningsudgaver)
    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)
    for temperatur in temperaturer:
        seed = None
        plt.figure(figsize=(10, 6))
        i = 0
        for j, fortræningsudgave in enumerate(fortræningsudgaver):
            runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave, seed), runs_in_group))
            rygrad_runids = set(list(map(viz0.get_rygrad_runid, runs_filtered)))
            df = viz0.get_df(runs_filtered)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='any')
            df = df.groupby('eftertræningsmængde').mean().reset_index()
            label = LABELLER[fortræningsudgave]
            plt.plot(df["eftertræningsmængde"][0:], df[f"test_loss_mean"][0:], color=farver[i], alpha=0.2)
            plt.scatter(df["eftertræningsmængde"][0:], df[f"test_loss_mean"][0:], color=farver[i], alpha=1.0, label=label)
            i += 1
        plt.title(f'{TITLER[temperatur]}', fontsize=22)
        plt.xlabel("Datamængde", fontsize=18)
        plt.ylabel("MAE", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.yscale("log")
        # plt.grid()
        plt.legend(fontsize=18)
        ax = plt.gca()
        # ax.get_yaxis().get_major_formatter().labelOnlyBase = False
        # ax.yaxis.set_major_locator(MultipleLocator(1000))  # Indstil større ticks hver 1 enhed
        # ax.yaxis.set_minor_locator(MultipleLocator(200))  # Indstil mindre ticks hver 0.1 enhed

        # Formatter for både major og minor ticks
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.jpg"))
        plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.pdf"))
        plt.close()
