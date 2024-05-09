import wandb
import shutil
import os
import matplotlib.pyplot as plt
from farver import farvekort, farver
import pandas as pd
import matplotlib.ticker as ticker

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

def main_filter(run, temperatur, mode, group):
    if run.group == None:
        return False
    if run.group != group:
        return False
    if mode not in run.tags:
        return False
    if mode in ['med', 'uden'] and temperatur not in run.tags:
        return False
    return True

if os.path.exists("eksp2_logs"):
    shutil.rmtree("eksp2_logs")

runs = wandb.Api().runs("afhandling")
runs = list(filter(is_downstream, runs))
runs = list(filter(has_suitable_metrics, runs))
groups = []
for run in runs:
    groups.append(run.group)
groups = list(set(groups))
groups = list(filter(is_eksp2, groups))
for group in groups:
    kørsel_path = os.path.join("eksp2_logs", group)
    os.makedirs(kørsel_path)
    for temperatur in ['frossen', 'optøet']:
        try:
            plt.figure(figsize=(10, 6))
            for i, mode in enumerate(['med', 'uden', 'baseline']):
                runs_filtered = list(filter(lambda w: main_filter(w, temperatur, mode, group), runs))
                df = get_df(runs_filtered)
                prefix = f'{mode}'
                plt.scatter(df["eftertræningsmængde"], df[f"test_loss_mean"], label=prefix, color=farver[i])
                plt.fill_between(df["eftertræningsmængde"], df[f"test_loss_lower"], df[f"test_loss_upper"],
                                 color=farver[i],
                                 alpha=0.3)
            plt.title(f'{group} {temperatur}')
            plt.xlabel("Datamængde")
            plt.ylabel("MAE")
            plt.yscale("log")
            plt.xticks(df["eftertræningsmængde"])
            plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=100))  # Juster 'numticks' for flere ticks
            plt.gca().yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
            # Brug LogFormatter for at vise tal ud for minor ticks også
            plt.gca().yaxis.set_minor_formatter(ticker.LogFormatter(base=10, labelOnlyBase=False))
            # plt.yticks([0.1, 1, 10, 100, 1000])
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(kørsel_path, f"{temperatur}.jpg"))
            plt.close()
        except ValueError:
            pass