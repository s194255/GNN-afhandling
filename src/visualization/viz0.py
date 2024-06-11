import pickle

import wandb
import pandas as pd
import json
import numpy as np
from src.redskaber import indlæs_yaml
import os
from tqdm import tqdm

METRICS = {'test_loss_mean', "test_loss_std", "test_loss_lower", "test_loss_upper", "eftertræningsmængde", "_runtime"}

JSON_KEYS = {'fortræningsudgave', 'temperatur'}

CACHE = "reports/cache"

def is_suitable(run, gruppenavn: str):
    group = run.group
    if group == None:
        return False
    if group.split("_")[0] != gruppenavn:
        return False
    metrics = set(run.summary.keys())
    if len(METRICS - metrics) != 0:
        return False
    json_keys = set(json.loads(run.json_config).keys())
    if len(JSON_KEYS - json_keys) != 0:
        return False
    return True

def get_group(run):
    return run.group

def get_groups_runs(gruppenavn):
    runs = wandb.Api().runs("afhandling")
    runs = list(filter(lambda w: is_suitable(w, gruppenavn), runs))
    grupper = sorted(set(list(map(get_group, runs))))
    return grupper, runs

def is_in_group(run, group):
    if run.group == None:
        return False
    else:
        return run.group == group


def get_fortræningsudgave(run):
    config = json.loads(run.json_config)
    return config['fortræningsudgave']['value']


def get_temperatur(run):
    config = json.loads(run.json_config)
    return config['temperatur']['value']


def get_seed(run):
    config = json.loads(run.json_config)
    return config['seed']['value']

def get_eftertræningsmængde(run):
    return run.summary['eftertræningsmængde']

def get_loops_params(group, runs):
    runs_in_group = list(filter(lambda w: is_in_group(w, group), runs))

    fortræningsudgaver_usorteret = set(list(map(get_fortræningsudgave, runs_in_group)))
    fortræningsudgaver = ['uden', '3D-EMGP-globalt', '3D-EMGP-begge', '3D-EMGP-lokalt', 'SelvvejledtQM9']
    fortræningsudgaver = [f for f in fortræningsudgaver if f in fortræningsudgaver_usorteret]

    temperaturer = set(list(map(get_temperatur, runs_in_group)))
    seeds = set(list(map(get_seed, runs_in_group)))
    rygrad_runids = set(list(map(get_rygrad_runid, runs_in_group)))
    return runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids

def get_eftertræningsmængder(group, runs):
    runs_in_group = list(filter(lambda w: is_in_group(w, group), runs))
    eftertræningsmængder = sorted(set(list(map(get_eftertræningsmængde, runs_in_group))))
    return eftertræningsmængder

def get_rygrad_runid(run):
    config = json.loads(run.json_config)
    return config['rygrad runid']['value']

def get_hidden_channels(run):
    config = json.loads(run.json_config)
    return config['args_dict']['value']['rygrad']['hidden_channels']

def main_filter(run, temperatur, fortræningsudgave, seed):
    run_temperatur = get_temperatur(run)
    if (run_temperatur != temperatur) and (temperatur is not None):
        return False
    if (get_fortræningsudgave(run) != fortræningsudgave) and (fortræningsudgave is not None):
        return False
    if (get_seed(run) != seed) and (seed is not None):
        return False
    return True

def kernel_baseline():
    x1, y1 = 100, 1.0*10**3
    x2, y2 = 20*10**3, 0.12*10**3
    a = (np.log(y2)-np.log(y1)) / (np.log(x2) - np.log(x1))
    b = y1/x1**a
    return lambda x: b*x**a

def get_df(runs):
    not_met_cols = ['seed', 'fortræningsudgave', 'temperatur']
    resultater = {nøgle: [] for nøgle in list(METRICS)+not_met_cols}
    resultater = pd.DataFrame(resultater)
    for run in runs:
        resultat = {nøgle: [værdi] for nøgle, værdi in run.summary.items() if nøgle in METRICS}
        resultat['seed'] = get_seed(run)
        resultat['fortræningsudgave'] = get_fortræningsudgave(run)
        resultat['temperatur'] = get_temperatur(run)
        resultat['hidden_channels'] = get_hidden_channels(run)

        resultater = pd.concat([resultater, pd.DataFrame(data=resultat)], ignore_index=True)
    sel_cols = [col for col in resultater.columns if col not in not_met_cols]
    resultater[sel_cols] = resultater[sel_cols].apply(pd.to_numeric, errors='coerce')
    resultater = resultater.dropna(how='any', subset=sel_cols)
    return resultater

def get_stjerner():
    ref = indlæs_yaml('reference_ckpts.yaml')
    return ref['eksp2']['frossen'] + ref['eksp2']['optøet']


def main_filter2(run, rygrad_runid):
    run_rygrad_runid = get_rygrad_runid(run)
    return run_rygrad_runid == rygrad_runid


FORT_LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9-fortræning',
            '3D-EMGP-lokalt': 'Lokalt',
            '3D-EMGP-globalt': 'Globalt',
            '3D-EMGP-begge': 'Begge'
            }


def get_group_df(group):
    if not os.path.exists(CACHE):
        os.makedirs(CACHE)
    cache_path = os.path.join(CACHE, f"{group}.pickle")
    if os.path.exists(cache_path):
        print("bruger cache")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        return cache['df']
    else:
        print("laver nyt cache")
        gruppenavn = group.split("_")[0]
        runs = wandb.Api().runs("afhandling")
        runs = list(filter(lambda w: is_suitable(w, gruppenavn), runs))
        runs_in_group, _, _, _, _ = get_loops_params(group, runs)
        df = get_df(runs_in_group)
        cache = {
            'df': df
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        return df

def get_loop_params_group_df(group_df):
    fortræningsudgaver_usorteret = group_df['fortræningsudgave'].unique()
    fortræningsudgaver = ['uden', '3D-EMGP-globalt', '3D-EMGP-begge', '3D-EMGP-lokalt', 'SelvvejledtQM9']
    fortræningsudgaver = [f for f in fortræningsudgaver if f in fortræningsudgaver_usorteret]

    temperaturer = group_df['temperatur'].unique()
    seeds = group_df['seed'].unique()
    return fortræningsudgaver, temperaturer, seeds
