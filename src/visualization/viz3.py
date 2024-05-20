import shutil
import os
import matplotlib.pyplot as plt
from farver import  farver
import pandas as pd
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
import viz0

TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'snydefortræning'}



if os.path.exists("eksp2_logs"):
    shutil.rmtree("eksp2_logs")

groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    if group not in ['eksp2_47', 'eksp2_48', 'eksp2_67']:
        continue
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    kørsel_path = os.path.join("eksp2_logs", group)
    os.makedirs(kørsel_path)
    for temperatur in temperaturer:
        plt.figure(figsize=(10, 6))
        for j, seed in enumerate(seeds):
            i = 0
            for fortræningsudgave in fortræningsudgaver:
                runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave, seed), runs_in_group))
                rygrad_runids = set(list(map(viz0.get_rygrad_runid, runs_filtered)))
                for rygrad_runid in rygrad_runids:
                    runs_filtered2 = list(filter(lambda w: viz0.main_filter2(w, rygrad_runid), runs_filtered))
                    df = viz0.get_df(runs_filtered2)
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(how='any')
                    if j == 0:
                        label = LABELLER[fortræningsudgave]
                    else:
                        label = None
                    plt.plot(df["eftertræningsmængde"], df[f"test_loss_mean"], label=label, color=farver[i])
                    plt.scatter(df["eftertræningsmængde"], df[f"test_loss_mean"], color=farver[i])
                    i += 1
        plt.title(f'{TITLER[temperatur]}', fontsize=22)
        plt.xlabel("Datamængde", fontsize=18)
        plt.ylabel("MAE ($m\\mathrm{a}_0^3$)", fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tick_params(axis='both', which='minor', labelsize=14)
        plt.yscale("log")
        plt.legend(fontsize=18)
        plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
        plt.savefig(os.path.join(kørsel_path, f"{temperatur}.jpg"))
        plt.savefig(os.path.join(kørsel_path, f"{temperatur}.pdf"))
        plt.close()
