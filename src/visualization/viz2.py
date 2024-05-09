import wandb
import shutil
import os
import matplotlib.pyplot as plt
from farver import farvekort, farver
import pandas as pd
import matplotlib.ticker as ticker
import json


METRICS = {'test_loss_mean', "test_loss_std", "test_loss_lower", "test_loss_upper", "eftertræningsmængde"}

JSON_KEYS = {'fortræningsudgave', 'temperatur'}

def get_group(run):
    return run.group

def get_fortræningsudgave(run):
    config = json.loads(run.json_config)
    return config['fortræningsudgave']['value']

def get_temperatur(run):
    config = json.loads(run.json_config)
    return config['temperatur']['value']

def is_suitable(run):
    group = run.group
    if group == None:
        return False
    if group.split("_")[0] != 'eksp2':
        return False
    metrics = set(run.summary.keys())
    if len(METRICS - metrics) != 0:
        return False
    json_keys = set(json.loads(run.json_config).keys())
    if len(JSON_KEYS - json_keys) != 0:
        return False
    return True

def is_in_group(run, group):
    if run.group == None:
        return False
    else:
        return run.group == group

def main_filter(run, temperatur, fortræningsudgave):
    run_temperatur = get_temperatur(run)
    if run_temperatur != temperatur:
        return False
    run_fortræningsudgave = get_fortræningsudgave(run)
    if run_fortræningsudgave != fortræningsudgave:
        return False
    # if 'baseline' in run.tags:
    #     return False
    return True

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
runs = list(filter(is_suitable, runs))

groups = set(list(map(get_group, runs)))
print(groups)
for group in groups:
    runs_group = list(filter(lambda w: is_in_group(w, group), runs))
    fortræningsudgaver = set(list(map(get_fortræningsudgave, runs_group)))
    print(fortræningsudgaver)
    kørsel_path = os.path.join("eksp2_logs", group)
    os.makedirs(kørsel_path)
    for temperatur in ['frossen']:
        try:
            plt.figure(figsize=(10, 6))
            for i, fortræningsudgave in enumerate(fortræningsudgaver):
                runs_filtered = list(filter(lambda w: main_filter(w, temperatur, fortræningsudgave), runs_group))
                df = get_df(runs_filtered)
                prefix = f'{fortræningsudgave}'
                plt.scatter(df["eftertræningsmængde"], df[f"test_loss_mean"], label=prefix, color=farver[i])
                plt.fill_between(df["eftertræningsmængde"], df[f"test_loss_lower"], df[f"test_loss_upper"],
                                 color=farver[i],
                                 alpha=0.3)
            plt.title(f'{group} {temperatur}')
            plt.xlabel("Datamængde")
            plt.ylabel("MAE")
            plt.yscale("log")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(kørsel_path, f"{temperatur}.jpg"))
            plt.close()
        except ValueError:
            pass
