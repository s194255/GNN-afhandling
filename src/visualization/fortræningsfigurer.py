import shutil
import os
import matplotlib.pyplot as plt
import wandb
import src.visualization.farver as far
# import pandas as pd
# from matplotlib.ticker import ScalarFormatter
# from tqdm import tqdm
# from src.visualization import viz0
# import numpy as np
import pickle
# import random




TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet rygrad"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9-fortræning',
            '3D-EMGP-lokalt': 'Lokalt',
            '3D-EMGP-globalt': 'Globalt',
            '3D-EMGP-begge': 'Begge'
            }


FIGNAVN = 'træningskurve'
ROOT = os.path.join('reports/figures/Eksperimenter/fortræning', FIGNAVN)

farveopslag = {
    '3D-EMGP-lokalt': far.bright_green,
    '3D-EMGP-globalt': far.blue,
    '3D-EMGP-begge': far.navy_blue,
    'SelvvejledtQM9': far.orange,
    'uden': far.corporate_red
}


STJERNER = {
    '3D-EMGP-lokalt': '8dxl194x',
    '3D-EMGP-globalt': 'u1eb0flg',
    '3D-EMGP-begge': 'eair1kzv',
    'SelvvejledtQM9': 'av6l2023'
}

CACHE = "reports/cache/fortræning"
# COLS = ['lr-AdamW', 'train_loss', 'val_loss']
COLS = ['train_loss', 'val_loss']
X_COL = 'trainer/global_step'

def get_group_dfs(runid):
    cache_path = os.path.join(CACHE, f'{runid}.pickle')
    if os.path.exists(cache_path):
        print("bruger cache")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        return cache['dfs']
    else:
        print("laver cache")
        runs = wandb.Api().runs("afhandling")
        has_run_id = lambda run: run.id == runid
        runs = list(filter(has_run_id, runs))
        print(runs)
        assert len(runs) == 1
        run = runs[0]
        dfs = {}
        for col in COLS:
            df = run.history(samples=10000, keys=[X_COL, col])
            dfs[col] = df
        dfs['epoch'] = run.history(samples=10000, keys=['epoch'])
        cache = {
            'dfs': dfs
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        return dfs



def plot(data: dict):
    fig, axs = plt.subplots(4, 2, figsize=(20, 25))  # Øget figurstørrelse for bedre plads
    titles = ['Træningstab', 'Valideringstab']  # Tilføjet titler for subplots

    for i, (fortræ, df_dict) in enumerate(data.items()):
        print(i)
        farve = farveopslag.get(fortræ, 'blue')  # Standardfarve hvis fortræ ikke findes i farveopslag
        for j, col in enumerate(COLS):
            ax = axs[i, j]
            df = df_dict[col]
            e_p_s = df_dict['epoch']['epoch'].max() / df[X_COL].max()
            df = df[[X_COL, col]].dropna(how='any')
            label = LABELLER[fortræ]
            ax.plot(df[X_COL]*e_p_s, df[col], color=farve, label=label)
            ax.set_ylabel('MAE', fontsize=35)
            ax.tick_params(axis='both', which='major', labelsize=35)
            ax.tick_params(axis='both', which='minor', labelsize=32)
            ax.grid(True)  # Tilføj grid for bedre læsbarhed
            ax.set_yscale('log')
            if j == 0:
                ax.legend(fontsize=45)

            if i == 0:
                ax.set_title(titles[j], fontsize=40)  # Tilføjelse af titler til subplots

            if i == len(data.items())-1:
                ax.set_xlabel('Epoke', fontsize=40)
            # else:
            #     ax.set_xticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    # fig.suptitle('Fortræningernes træningsmetrikker', fontsize=45)  # Tilføj hovedtitel
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{FIGNAVN}.pdf"))
    plt.legend()
    plt.close()



if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

if not os.path.exists(CACHE):
    os.makedirs(CACHE)

kørsel_path = os.path.join(ROOT)
os.makedirs(kørsel_path)

data = {}
for fortræ, runid in STJERNER.items():
    group_dfs = get_group_dfs(runid)
    data[fortræ] = group_dfs
plot(data)


# dfs = {}
# for temperatur in temperaturer:
#     group = stjerner[temperatur]
#     runs_in_group, fortræningsudgaver, temperaturer_lp, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
#     runs_in_group = list(filter(lambda run: viz0.get_eftertræningsmængde(run) == 500, runs_in_group))
#     run = random.choice(runs_in_group)
#     dfs[temperatur] = run.history()
# plot(dfs)