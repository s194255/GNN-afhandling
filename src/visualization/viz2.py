import shutil
import os
import matplotlib.pyplot as plt
from farver import farver
import viz0
import pandas as pd
from tqdm import tqdm

if os.path.exists("eksp2_logs"):
    shutil.rmtree("eksp2_logs")


groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    kørsel_path = os.path.join("eksp2_logs", group)
    os.makedirs(kørsel_path)
    for temperatur in temperaturer:
        for seed in seeds:
                plt.figure(figsize=(10, 6))
                i = 0
                for fortræningsudgave in fortræningsudgaver:
                    runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave, seed), runs_in_group))
                    df = viz0.get_df(runs_filtered)
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(how='any')
                    prefix = f'{fortræningsudgave}'
                    plt.scatter(df["eftertræningsmængde"], df[f"test_loss_mean"], label=prefix, color=farver[i])
                    plt.fill_between(df["eftertræningsmængde"], df[f"test_loss_lower"], df[f"test_loss_upper"],
                                     color=farver[i],
                                     alpha=0.3)
                    i += 1
                plt.title(f'{group} {temperatur} {seed}')
                plt.xlabel("Datamængde")
                plt.ylabel("MAE")
                plt.yscale("log")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{seed}.jpg"))
                plt.close()
