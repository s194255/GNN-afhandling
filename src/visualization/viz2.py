import wandb
import shutil
import os
import matplotlib.pyplot as plt
from farver import farvekort, farver
import pandas as pd

METRICS = {'test_loss_mean', "test_loss_std", "test_loss_lower", "test_loss_upper", "eftertræningsmængde"}

def is_eksp2(group):
    if group == None:
        return False
    return group.split("_")[0] == 'eksp2'

def is_in_group(run, group):
    # group = run.group
    if run.group == None:
        return False
    else:
        return run.group == group

def is_downstream(run):
    return 'downstream' in run.tags

def has_suitable_metrics(run):
    metrics = set(run.summary.keys())
    return len(METRICS - metrics) == 0

def rygrad_temperatur(run, temperatur):
    return temperatur in run.tags

def has_mode(run, mode):
    return mode in run.tags

def get_df(runs):
    resultater = {nøgle: [] for nøgle in METRICS}
    resultater = pd.DataFrame(resultater)
    for run in runs:
        resultat = {nøgle: [værdi] for nøgle, værdi in run.summary.items() if nøgle in METRICS}
        resultater = pd.concat([resultater, pd.DataFrame(data=resultat)], ignore_index=True)
    return resultater

if os.path.exists("eksp2_logs"):
    shutil.rmtree("eksp2_logs")

runs = wandb.Api().runs("afhandling")
print(len(runs))
runs = list(filter(is_downstream, runs))
runs = list(filter(has_suitable_metrics, runs))
print(len(runs))
groups = []
# runs = []
for run in runs:
    groups.append(run.group)
groups = list(set(groups))
groups = list(filter(is_eksp2, groups))
print(groups)
for group in groups:
    run_group = list(filter(lambda w: is_in_group(w, group), runs))
    kørsel_path = os.path.join("eksp2_logs", group)
    os.makedirs(kørsel_path)
    print(run_group)
    for temperatur in ['frossen', 'optøet']:
        run_group_temperatur = list(filter(lambda w: rygrad_temperatur(w, temperatur), run_group))
        plt.figure(figsize=(10, 6))
        for i, mode in enumerate(['med', 'uden']):
            run_group_temperatur_opgave = list(filter(lambda w: has_mode(w, mode), run_group_temperatur))
            df = get_df(run_group_temperatur_opgave)
            prefix = f'{mode}_{temperatur}'
            plt.scatter(df["eftertræningsmængde"], df[f"test_loss_mean"], label=prefix, color=farver[i])
            plt.fill_between(df["eftertræningsmængde"], df[f"test_loss_lower"], df[f"test_loss_upper"],
                             color=farver[i],
                             alpha=0.3)
        plt.title(f'{group} {temperatur}')

        plt.xlabel("Datamængde")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(kørsel_path, f"{temperatur}.jpg"))
        plt.close()