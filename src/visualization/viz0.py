import pickle
from itertools import product

import wandb
import pandas as pd
import json
import numpy as np
# from src.redskaber import indlæs_yaml
import os
import yaml

from src.visualization import farver as far


def indlæs_yaml(sti):
    with open(sti, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

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

def get_num_layers(run):
    config = json.loads(run.json_config)
    return config['args_dict']['value']['hoved']['num_layers']

def get_predicted_attribute(run):
    config = json.loads(run.json_config)
    try:
        return config['args_dict']['value']['predicted_attribute']
    except KeyError:
        return 'MD17'

def get_weight_decay(run):
    config = json.loads(run.json_config)
    return config['args_dict']['value']['weight_decay']

def get_name(run):
    config = json.loads(run.json_config)
    return config['name']['value']

def main_filter(run, temperatur, fortræningsudgave, seed):
    run_temperatur = get_temperatur(run)
    if (run_temperatur != temperatur) and (temperatur is not None):
        return False
    if (get_fortræningsudgave(run) != fortræningsudgave) and (fortræningsudgave is not None):
        return False
    if (get_seed(run) != seed) and (seed is not None):
        return False
    return True

def kernel_baseline(predicted_attribute):
    if predicted_attribute == 0:
        x1, y1 = 101.85, 1.23 * 10 ** 3
        x2, y2 = 19277.496, 0.767 * 10 ** 3
    elif predicted_attribute == 1:
        x1, y1 = 100, 1.16 * 10 ** 3
        x2, y2 = 19277.496, 0.152 * 10 ** 3
    else:
        raise NotImplementedError
    a = (np.log(y2)-np.log(y1)) / (np.log(x2) - np.log(x1))
    b = y1/x1**a
    return lambda x: b*x**a

def get_df(runs):
    nan_allowed_cols = ['seed', 'fortræningsudgave', 'temperatur',
                        'rygrad runid', 'predicted_attribute', 'name']
    resultater = {nøgle: [] for nøgle in list(METRICS)+nan_allowed_cols}
    resultater = pd.DataFrame(resultater)
    for run in runs:
        resultat = {nøgle: [værdi] for nøgle, værdi in run.summary.items() if nøgle in METRICS}
        resultat['seed'] = get_seed(run)
        resultat['fortræningsudgave'] = get_fortræningsudgave(run)
        resultat['temperatur'] = get_temperatur(run)
        resultat['hidden_channels'] = get_hidden_channels(run)
        resultat['num_layers'] = get_num_layers(run)
        resultat['predicted_attribute'] = get_predicted_attribute(run)
        resultat['weight_decay'] = get_weight_decay(run)
        resultat['rygrad runid'] = get_rygrad_runid(run)
        resultat['name'] = get_name(run)

        resultater = pd.concat([resultater, pd.DataFrame(data=resultat)], ignore_index=True)
    nan_illegal_cols = [col for col in resultater.columns if col not in nan_allowed_cols]
    resultater[nan_illegal_cols] = resultater[nan_illegal_cols].apply(pd.to_numeric, errors='coerce')
    resultater = resultater.dropna(how='any', subset=nan_illegal_cols)
    return resultater

def get_stjerner():
    ref = indlæs_yaml('reference_ckpts.yaml')
    return ref['eksp2']['frossen'] + ref['eksp2']['optøet']


def main_filter2(run, rygrad_runid):
    run_rygrad_runid = get_rygrad_runid(run)
    return run_rygrad_runid == rygrad_runid


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
        runs = wandb.Api(timeout=290).runs("afhandling")
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


def set_size(fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------

    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    textwidth = 426.79135

    # Width of figure (in pts)
    fig_width_pt = textwidth * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


FARVEOPSLAG = {
    '3D-EMGP-lokalt': far.green,
    '3D-EMGP-globalt': far.blue,
    '3D-EMGP-begge': far.navy_blue,
    'SelvvejledtQM9': far.orange,
    'uden': far.corporate_red,
}

FORT_LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9-fortræning',
            '3D-EMGP-lokalt': 'Lokal',
            '3D-EMGP-globalt': 'Global',
            '3D-EMGP-begge': 'Lokal+Global'
            }


def sanity_check_group_df(group_df):
    num_seeds = len(group_df['seed'].unique())
    fortræer = group_df['fortræningsudgave'].unique()
    datamængder = group_df['eftertræningsmængde'].unique()
    seeds = group_df['seed'].unique()
    for fortræ, datamængde in product(fortræer, datamængder):
        idxs = group_df['fortræningsudgave'] == fortræ
        idxs = (idxs) & (group_df['eftertræningsmængde'] == datamængde)
        if len(group_df[idxs]) != num_seeds:
            print(f"fortræ = {fortræ}, datamængde = {datamængde}  er ikke ok. Den har {len(group_df[idxs])} frø!")
            duplicates = group_df[idxs]['seed'].value_counts()
            duplicates = duplicates[duplicates > 1].index
            print("Gengangere i serien:", duplicates)
            print("\n")
        else:
            print(f"fortræ = {fortræ}, datamængde = {datamængde}  er ok")
            print("\n")
    for seed in seeds:
        forv_num_seeds = len(fortræer)*len(datamængder)
        if sum(group_df['seed'] == seed) != forv_num_seeds:
            print(f"FEJL!!! seed = {seed} er IKKE ok FEJL!!")
        else:
            print(f"seed = {seed} er ok")
        # assert len(group_df[idxs]) == 33
