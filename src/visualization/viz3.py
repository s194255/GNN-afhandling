import wandb
import shutil
import os
import matplotlib.pyplot as plt
from farver import farvekort, farver
import pandas as pd
import matplotlib.ticker as ticker
import json
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter


METRICS = {'test_loss_mean', "test_loss_std", "test_loss_lower", "test_loss_upper", "eftertræningsmængde"}

JSON_KEYS = {'fortræningsudgave', 'temperatur'}

TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP'}

def get_group(run):
    return run.group

def get_fortræningsudgave(run):
    config = json.loads(run.json_config)
    return config['fortræningsudgave']['value']

def get_temperatur(run):
    config = json.loads(run.json_config)
    return config['temperatur']['value']

def get_rygrad_runid(run):
    config = json.loads(run.json_config)
    return config['rygrad runid']['value']

def get_seed(run):
    config = json.loads(run.json_config)
    return config['seed']['value']

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

def main_filter(run, temperatur, fortræningsudgave, seed):
    run_temperatur = get_temperatur(run)
    if run_temperatur != temperatur:
        return False
    run_fortræningsudgave = get_fortræningsudgave(run)
    if run_fortræningsudgave != fortræningsudgave:
        return False
    if seed != get_seed(run):
        return False
    return True

def main_filter2(run, rygrad_runid):
    run_rygrad_runid = get_rygrad_runid(run)
    return run_rygrad_runid == rygrad_runid

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
for group in tqdm(groups):
    if group not in ['eksp2_47', 'eksp2_48']:
        continue
    runs_group = list(filter(lambda w: is_in_group(w, group), runs))
    fortræningsudgaver = set(list(map(get_fortræningsudgave, runs_group)))
    temperaturer = set(list(map(get_temperatur, runs_group)))
    seeds = set(list(map(get_seed, runs_group)))
    kørsel_path = os.path.join("eksp2_logs", group)
    os.makedirs(kørsel_path)
    for temperatur in temperaturer:
        try:
            plt.figure(figsize=(10, 6))
            for j, seed in enumerate(seeds):
                i = 0
                for fortræningsudgave in fortræningsudgaver:
                    runs_filtered = list(filter(lambda w: main_filter(w, temperatur, fortræningsudgave, seed), runs_group))
                    rygrad_runids = set(list(map(get_rygrad_runid, runs_filtered)))
                    for rygrad_runid in rygrad_runids:
                        runs_filtered2 = list(filter(lambda w: main_filter2(w, rygrad_runid), runs_filtered))
                        df = get_df(runs_filtered2)
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
        except ValueError:
            pass
