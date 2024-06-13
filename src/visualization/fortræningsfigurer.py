import shutil
import os
import matplotlib.pyplot as plt
import wandb
import src.visualization.farver as far
import pandas as pd
from matplotlib.ticker import ScalarFormatter
# from tqdm import tqdm
# from src.visualization import viz0
import numpy as np
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
COLS = ['train_loss', 'val_loss', 'train_lokalt_loss', 'train_globalt_loss']
FORTRÆ_COLS  = {
    '3D-EMGP-lokalt': ['train_lokalt_loss', 'val_loss'],
    '3D-EMGP-globalt': ['train_globalt_loss', 'val_loss'],
    '3D-EMGP-begge': ['train_lokalt_loss', 'train_globalt_loss', 'train_loss', 'val_loss'],
    'SelvvejledtQM9': ['train_loss', 'val_loss']
}
COL_TITEL = {
    'train_loss': 'Træningstab (samlet)',
    'val_loss': 'Valideringstab',
    'train_lokalt_loss': 'Lokalt tab',
    'train_globalt_loss': 'Globalt tab',
}
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


def make_table(data: dict):
    new_df = {col: [] for col in COLS}
    new_df = {**{'fortræningsudgave': []}, **new_df}
    new_df = pd.DataFrame(data=new_df)
    for fortræ, df_dict in data.items():
        new_df_linje = {'fortræningsudgave': [fortræ]}
        for col in COLS:
            if fortræ == 'SelvvejledtQM9' and col in ['train_lokalt_loss', 'train_globalt_loss']:
                continue
            df = df_dict[col]
            df[col] = df[col].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='any')
            try:
                new_df_linje[col] = [df[col].min()]
            except TypeError:
                a = 2
        new_df = pd.concat([new_df, pd.DataFrame(data=new_df_linje)], ignore_index=True)


def plot(data: dict):
    # fig, axs = plt.subplots(4, len(COLS), figsize=(40, 30))  # Øget figurstørrelse for bedre plads
    # titles = ['Træningstab', 'Valideringstab']  # Tilføjet titler for subplots

    for i, (fortræ, df_dict) in enumerate(data.items()):
        farve = farveopslag.get(fortræ, 'blue')  # Standardfarve hvis fortræ ikke findes i farveopslag
        cols = FORTRÆ_COLS[fortræ]
        fig, axs = plt.subplots(1, len(cols), figsize=(40, 10))
        for j, col in enumerate(cols):
            if fortræ == 'SelvvejledtQM9' and col in ['train_lokalt_loss', 'train_globalt_loss']:
                continue
            ax = axs[j]
            df = df_dict[col]
            e_p_s = df_dict['epoch']['epoch'].max() / df[X_COL].max()
            df = df[[X_COL, col]].dropna(how='any')
            label = LABELLER[fortræ]
            ax.plot(df[X_COL]*e_p_s, df[col], color=farve, label=label)
            if j == 0:
                ax.set_ylabel('MAE', fontsize=50)
            ax.tick_params(axis='both', which='major', labelsize=35)
            ax.tick_params(axis='both', which='minor', labelsize=30)
            ax.grid(True)  # Tilføj grid for bedre læsbarhed
            ax.set_yscale('log')
            ax.set_xscale('log')
            # ax.xaxis.set_minor_formatter(ScalarFormatter())
            # ax.xaxis.set_major_formatter(ScalarFormatter())

            max_epoch = df[X_COL].max() * e_p_s
            xticks = np.logspace(0, np.log10(max_epoch), num=5)  # Adjust the number of ticks with 'num'
            rounded_xticks = np.round(xticks).astype(int)
            ax.set_xticks(rounded_xticks)
            ax.set_xticklabels(rounded_xticks)



            # ax.set_xticklabels(np.linspace(0, 50, 10))
            if j == 0:
                ax.legend(fontsize=50)

            titel = COL_TITEL[col]
            ax.set_title(titel, fontsize=50)  # Tilføjelse af titler til subplots
            ax.set_xlabel('Epoke', fontsize=50)

        plt.tight_layout()
        # plt.subplots_adjust(top=0.92)
        # fig.suptitle('Fortræningernes træningsmetrikker', fontsize=45)  # Tilføj hovedtitel
        plt.savefig(os.path.join(kørsel_path, f"{fortræ}.jpg"))
        plt.savefig(os.path.join(kørsel_path, f"{fortræ}.pdf"))
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
# make_table(data)


# dfs = {}
# for temperatur in temperaturer:
#     group = stjerner[temperatur]
#     runs_in_group, fortræningsudgaver, temperaturer_lp, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
#     runs_in_group = list(filter(lambda run: viz0.get_eftertræningsmængde(run) == 500, runs_in_group))
#     run = random.choice(runs_in_group)
#     dfs[temperatur] = run.history()
# plot(dfs)