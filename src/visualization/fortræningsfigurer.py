import shutil
import os
import matplotlib.pyplot as plt
import wandb
import src.visualization.farver as far
import pandas as pd
from matplotlib.ticker import ScalarFormatter
# from tqdm import tqdm
from src.visualization import viz0
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

STJERNER_OPSLAG = {
    'QM9': {
        '3D-EMGP-lokalt': 'd4e2ftz6',
        '3D-EMGP-globalt': 'yhxjxrvk',
        '3D-EMGP-begge': 'gabq3exm',
        'SelvvejledtQM9': 'uvv3hn84'
    },
    'MD17': {
        '3D-EMGP-lokalt': 'ig3v4c4k',
        '3D-EMGP-globalt': 'z4au4jjl',
        '3D-EMGP-begge': 'esk3r63t',
    }
}

CACHE = "reports/cache/fortræning"
# COLS = ['lr-AdamW', 'train_loss', 'val_loss']
# COLS = ['train_loss', 'val_loss', 'train_lokalt_loss', 'train_globalt_loss']
COLS = ['train_lokalt_loss', 'train_globalt_loss', 'train_loss', 'val_loss']
FORTRÆ_COLS  = {
    '3D-EMGP-lokalt': ['train_lokalt_loss', 'val_loss'],
    '3D-EMGP-globalt': ['train_globalt_loss', 'val_loss'],
    '3D-EMGP-begge': ['train_lokalt_loss', 'train_globalt_loss', 'train_loss', 'val_loss'],
    'SelvvejledtQM9': ['train_loss', 'val_loss']
}
# COL_TITEL = {
#     'train_loss': r'$\left( \mathcal{D}_T, \mathcal{L}_{\text{lokal}} + \mathcal{L}_{\text{global}} \right)$',
#     'val_loss': r'$\left( \mathcal{D}_V, \mathcal{L}_{\text{lokal}} + \mathcal{L}_{\text{global}} \right)$',
#     'train_lokalt_loss': r'$\left( \mathcal{D}_T,\mathcal{L}_{\text{lokal}} \right)$',
#     'train_globalt_loss': r'$\left( \mathcal{D}_T,\mathcal{L}_{\text{global}} \right)$',
# }
COL_TITEL = {
    'train_loss': r'$\mathcal{L}_{EMGP}$ på For-træn-datasilo',
    'val_loss': r'$\mathcal{L}_{EMGP}$ på For-val-datasilo',
    'train_lokalt_loss': r'$\mathcal{L}_{lokal}$ på For-træn-datasilo',
    'train_globalt_loss': r'$\mathcal{L}_{global}$ på For-træn-datasilo',
}
X_COL = 'trainer/global_step'

fortræ_to_nrows = {
    '3D-EMGP-lokalt': 1,
    '3D-EMGP-globalt': 1,
    '3D-EMGP-begge': 2,
    'SelvvejledtQM9': 1,
}
fortræ_to_figsize = {
    '3D-EMGP-lokalt': (40, 10),
    '3D-EMGP-globalt': (40, 10),
    '3D-EMGP-begge': (40, 20),
    'SelvvejledtQM9': (40, 10),
}


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
            try:
                if len(df) != len(df[[X_COL, col]].dropna(how='any')):
                    print("nan fundet under fremstilling af cache")
            except KeyError:
                pass
        dfs['epoch'] = run.history(samples=10000, keys=['epoch'])
        cache = {
            'dfs': dfs
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        return dfs


# def plot_nan(df: pd.DataFrame, ax: plt.Axes):


def make_table(data: dict, dataset: str):
    cutoff = 150
    h = lambda col: f'{COL_TITEL[col]}'.lower()
    new_df = {h(col): [] for col in COLS}
    new_df = {**{'fortræningsudgave': []}, **new_df}
    new_df = pd.DataFrame(data=new_df)
    for fortræ, df_dict in data.items():
        fortræningsudgave = viz0.FORT_LABELLER[fortræ]
        new_df_linje = {'fortræningsudgave': [fortræningsudgave]}
        for col in COLS:
            if fortræ == 'SelvvejledtQM9' and col in ['train_lokalt_loss', 'train_globalt_loss']:
                continue
            df = df_dict[col]
            e_p_s = df_dict['epoch']['epoch'].max() / df[X_COL].max()
            df.drop(df[df[X_COL] > cutoff * e_p_s ** (-1)].index, inplace=True)
            df[col] = df[col].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='any')
            try:
                new_df_linje[h(col)] = [df[col].min()]
            except TypeError:
                pass
        new_df = pd.concat([new_df, pd.DataFrame(data=new_df_linje)], ignore_index=True)

    formatted_df = new_df.map(lambda x: f'{x:.2f}' if isinstance(x, (float, int)) else x)
    # formatted_df['fortræningsudgave'] = formatted_df['fortræningsudgave'].apply(lambda x: f'\\textbf{{{x}}}')
    latex_table = formatted_df.to_latex(index=False, escape=False, column_format='l|cccc')
    lines = latex_table.splitlines()
    lines.insert(2, '\\midrule')
    latex_table = '\n'.join(lines)

    latex_table = f"\\begin{{small}}\n{latex_table}\n\\end{{small}}"
    with open(os.path.join(kørsel_path, "minima.tex"), "w", encoding='utf-8') as f:
        f.write(latex_table)

    print(dataset)
    print(formatted_df.to_latex(index=False))
    print("\n")


def plot(data: dict, datasæt: str):
    cutoff = 150
    # plt.rcParams['text.usetex'] = True
    for i, (fortræ, df_dict) in enumerate(data.items()):
        farve = farveopslag.get(fortræ, 'blue')
        cols = FORTRÆ_COLS[fortræ]
        fig, axs = plt.subplots(fortræ_to_nrows[fortræ], 2, figsize=fortræ_to_figsize[fortræ])
        axs = axs.ravel()
        for j, col in enumerate(cols):
            if fortræ == 'SelvvejledtQM9' and col in ['train_lokalt_loss', 'train_globalt_loss']:
                continue
            ax = axs[j]
            df = df_dict[col]
            e_p_s = df_dict['epoch']['epoch'].max() / df[X_COL].max()
            df.drop(df[df[X_COL] > cutoff*e_p_s**(-1)].index, inplace=True)

            df[col] = df[col].apply(pd.to_numeric, errors='coerce')

            col_intp = f'col_intp'
            nan_indices = df[col].isna()
            df[col_intp] = df[col].interpolate()
            if nan_indices.sum() > 0:
                ax.scatter(df[X_COL][nan_indices]*e_p_s, df[col_intp][nan_indices],
                           color='#f4151a', marker='x', s=150)
            label = viz0.FORT_LABELLER[fortræ]
            ax.plot(df[X_COL] * e_p_s, df[col], color=farve, label=label)
            if j % 2 == 0:
                ax.set_ylabel('MAE', fontsize=50)
            ax.grid(True)  # Tilføj grid for bedre læsbarhed
            ax.set_yscale('log')
            ax.set_xscale('log')

            ax.tick_params(axis='both', which='major', labelsize=45)
            ax.tick_params(axis='both', which='minor', labelsize=40)
            ax.tick_params(axis='x', labelrotation=45)
            max_epoch = df[X_COL].max() * e_p_s
            xticks = np.logspace(0, np.log10(max_epoch), num=5)  # Adjust the number of ticks with 'num'
            rounded_xticks = np.round(xticks).astype(int)
            ax.set_xticks(rounded_xticks)
            ax.set_xticklabels(rounded_xticks)
            y_ticks = np.geomspace(df[col].min(), df[col].max(), num=5)
            ax.set_yticks(y_ticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()


            # ax.set_xticklabels(np.linspace(0, 50, 10))
            if j == 0:
                ax.legend(fontsize=50)

            titel = COL_TITEL[col]
            ax.set_title(titel, fontsize=60)  # Tilføjelse af titler til subplots
            ax.set_xlabel('Epoke', fontsize=50)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        fig.suptitle(datasæt, fontsize=70)  # Tilføj hovedtitel
        for ext in ['jpg', 'pdf']:
            if not os.path.exists(os.path.join(kørsel_path, f'{ext}s')):
                os.mkdir(os.path.join(kørsel_path, f'{ext}s'))
            plt.savefig(os.path.join(kørsel_path, f'{ext}s', f"{datasæt}_{fortræ}.{ext}"))
        plt.legend()
        plt.close()



if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

if not os.path.exists(CACHE):
    os.makedirs(CACHE)

kørsel_path = os.path.join(ROOT)
os.makedirs(kørsel_path)


for dataset, stjerner in STJERNER_OPSLAG.items():
    data = {}
    for fortræ, runid in stjerner.items():
        group_dfs = get_group_dfs(runid)
        data[fortræ] = group_dfs
    plot(data, dataset)
    make_table(data, dataset)
