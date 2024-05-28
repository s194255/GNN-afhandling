import wandb
import pandas as pd
import json

METRICS = {'test_loss_mean', "test_loss_std", "test_loss_lower", "test_loss_upper", "eftertræningsmængde"}

JSON_KEYS = {'fortræningsudgave', 'temperatur'}

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
    fortræningsudgaver = sorted(set(list(map(get_fortræningsudgave, runs_in_group))))
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


def main_filter(run, temperatur, fortræningsudgave, seed):
    run_temperatur = get_temperatur(run)
    if (run_temperatur != temperatur) and (temperatur is not None):
        return False
    if (get_fortræningsudgave(run) != fortræningsudgave) and (fortræningsudgave is not None):
        return False
    if (get_seed(run) != seed) and (seed is not None):
        return False
    return True


def get_df(runs):
    resultater = {nøgle: [] for nøgle in METRICS}
    resultater = pd.DataFrame(resultater)
    for run in runs:
        resultat = {nøgle: [værdi] for nøgle, værdi in run.summary.items() if nøgle in METRICS}
        resultat['seed'] = get_seed(run)
        resultat['fortræningsudgave'] = get_fortræningsudgave(run)

        resultater = pd.concat([resultater, pd.DataFrame(data=resultat)], ignore_index=True)
    return resultater, ['seed', 'fortræningsudgave']


def main_filter2(run, rygrad_runid):
    run_rygrad_runid = get_rygrad_runid(run)
    return run_rygrad_runid == rygrad_runid
