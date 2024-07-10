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

ROOT = 'reports/figures/Eksperimenter/2/ren'

udvalgte = None


if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    if udvalgte != None:
        if group not in [f'eksp2_{udvalgt}' for udvalgt in udvalgte]:
            continue

    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)
    for temperatur in temperaturer:
        plt.figure(figsize=(10, 6))
        i = 0
        for fortræningsudgave in fortræningsudgaver:
            runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave, None), runs_in_group))
            rygrad_runids = set(list(map(viz0.get_rygrad_runid, runs_filtered)))
            for rygrad_runid in rygrad_runids:
                for j, seed in enumerate(seeds):
                    runs_filtered = list(
                        filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave, seed), runs_in_group))
                    runs_filtered2 = list(filter(lambda w: viz0.main_filter2(w, rygrad_runid), runs_filtered))
                    df, ignore_columns = viz0.get_df(runs_filtered2)
                    sel_cols = [col for col in df.columns if col not in ignore_columns]
                    df = df[sel_cols].apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(how='any')
                    label = LABELLER[fortræningsudgave] if j == 0 else None
                    plt.plot(df["eftertræningsmængde"], df[f"test_loss_mean"], color=farver[i], alpha=0.2)
                    plt.scatter(df["eftertræningsmængde"], df[f"test_loss_mean"], color=farver[i], alpha=1.0, label=label)
                i += 1
        plt.title(f'{TITLER[temperatur]}', fontsize=22)
        plt.xlabel("Datamængde", fontsize=18)
        plt.ylabel("MAE", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tick_params(axis='both', which='minor', labelsize=14)
        plt.yscale("log")
        plt.legend(fontsize=18)
        plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.savefig(os.path.join(kørsel_path, f"{temperatur}.jpg"))
        plt.savefig(os.path.join(kørsel_path, f"{temperatur}.pdf"))
        plt.close()
