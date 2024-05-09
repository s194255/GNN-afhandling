import wandb
import shutil
import os
import matplotlib.pyplot as plt
from farver import farvekort, farver
import pandas as pd
import matplotlib.ticker as ticker


METRICS = {'test_loss_mean', "test_loss_std", "test_loss_lower", "test_loss_upper", "eftertræningsmængde"}
def get_group(run):
    return run.group

def is_suitable(run):
    group = run.group
    if group == None:
        return False
    if group.split("_")[0] != 'eksp2':
        return False
    metrics = set(run.summary.keys())
    if len(METRICS - metrics) != 0:
        return False
    return True

def is_in_group(run, group):
    if run.group == None:
        return False
    else:
        return run.group == group

if os.path.exists("eksp2_logs"):
    shutil.rmtree("eksp2_logs")

runs = wandb.Api().runs("afhandling")
runs = list(filter(is_suitable, runs))

groups = set(list(map(get_group, runs)))
print(groups)
for group in groups:
    runs_group = list(filter(lambda w: is_in_group(w, group), runs))
    kørsel_path = os.path.join("eksp2_logs", group)
    os.makedirs(kørsel_path)
