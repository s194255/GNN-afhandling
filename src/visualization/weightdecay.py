import pandas as pd

from src.visualization import viz0
import numpy as np
import os
import pickle
from itertools import product

def kaos():
    group_df = viz0.get_group_df('kaos_0')
    seeds = group_df['seed'].unique()
    fortræer = group_df['fortræningsudgave'].unique()
    col = 'test_loss_mean'
    for seed, fortræ in product(seeds, fortræer):
        idxs = group_df['fortræningsudgave'] == fortræ
        idxs = (idxs) & (group_df['seed'] == seed)

        mean = group_df[idxs][col].mean()
        std = group_df[idxs][col].std()
        n = len(group_df[idxs][col])
        print(f"fortræ = {fortræ}")
        print(f"seed = {seed}")
        print(f"mean = {mean}")
        print(f"std = {std}")
        print(f"std/mean = {std / mean}")
        print(f"n = {n}")
        print("\n")

    def compare_groups(group):
        uden = group[group['fortræningsudgave'] == 'uden'][col].item()
        med = group[group['fortræningsudgave'] == '3D-EMGP-begge'][col].item()
        return med < uden

    wins = group_df.groupby('seed').apply(compare_groups, include_groups=False)

    # Beregn gennemsnittet af resultaterne
    mean_wins = wins.mean()
    print(mean_wins)

    # wins = []
    # for seed in seeds:
    #     idxs = group_df['seed'] == seed
    #
    #     uden = (idxs) & (group_df['fortræningsudgave'] == 'uden')
    #     med = (idxs) & (group_df['fortræningsudgave'] == '3D-EMGP-begge')
    #     wins.append(group_df[med][col].item() < group_df[uden][col].item())
    # print(np.mean(np.array(wins)))


def nedtaktLilleVsStort():
    group_df = viz0.get_group_df('StortVsLille_1')
    fortræer = group_df['fortræningsudgave'].unique()
    hs = sorted(group_df['hidden_channels'].unique())
    col = 'test_loss_mean'
    for h, fortræ in product(hs, fortræer):
        idxs = group_df['fortræningsudgave'] == fortræ
        idxs = (idxs) & (group_df['hidden_channels'] == h)
        mean = group_df[idxs][col].mean()
        std = group_df[idxs][col].std()
        n = len(group_df[idxs][col])
        print(f"fortræ = {fortræ}")
        print(f"hidden channels = {h}")
        print(f"mean = {mean}")
        print(f"std = {std}")
        print(f"std/mean = {std/mean}")
        print(f"n = {n}")
        print("\n")


def regVsClass():
    runid_to_loss = {
        '7cajugge': 'class',
        'x96gc9d0': 'reg'
    }
    group_df = viz0.get_group_df('RegVsClass_1')
    for k, v in runid_to_loss.items():
        ixs = group_df['rygrad runid'] == k
        mean = group_df[ixs]['test_loss_mean'].mean()
        std = group_df[ixs]['test_loss_mean'].std()
        print(f"loss = {v}")
        print(f"mean = {mean}")
        print(f"std = {std}")
        print(f"std/mean = {std/mean}")
        print("\n")

def klogtVsDumt():
    runid_to_loss = {
        'a0zjuidn': 'klogt',
        '219rx1vj': 'dumt'
    }
    group_df = viz0.get_group_df('KlogtVsDumt_1')
    for k, v in runid_to_loss.items():
        ixs = group_df['rygrad runid'] == k
        mean = group_df[ixs]['test_loss_mean'].mean()
        std = group_df[ixs]['test_loss_mean'].std()
        n = len(group_df[ixs])
        print(f"loss = {v}")
        print(f"mean = {mean}")
        print(f"std = {std}")
        print(f"std/mean = {std / mean}")
        print(f"n = {n}")
        print("\n")

def weight_decay2():
    group_df = viz0.get_group_df('eksp4_2')
    weightdecays = sorted(group_df['weight_decay'].unique())
    for weightdecay in weightdecays:
        idxs = group_df['weight_decay'] == weightdecay
        mean = group_df[idxs]['test_loss_mean'].mean()
        std = group_df[idxs]['test_loss_mean'].std()
        n = len(group_df[idxs])
        print(f"weight decay = {np.log10(weightdecay)}")
        print(f"mean = {mean}")
        print(f"std = {std}")
        print(f"n = {n}")
        print("\n")


# kørselid = [0, 1, 2, 3, 4, 5, 6, 7]
# groups = [f'weightdecay_{k}' for k in kørselid]

# create_cache = False
#
# # prefix = 'støjniveau'
# # kørselid = [0, 1, 2]
# prefix = 'weightdecay'
# kørselid = [0, 1, 2, 3, 4, 5, 6, 7]
#
#
# groups = [f'{prefix}_{k}' for k in kørselid]
# group_dfs = []
# for k, group in zip(kørselid, groups):
#     group_df = viz0.get_group_df(group)
#     weightdecay = group_df['weight_decay'].unique()
#     print(k)
#     print(np.log10(weightdecay))
#     mean = group_df['test_loss_mean'].mean()
#     std = group_df['test_loss_mean'].std()
#     print(f"mean = {mean}")
#     print(f"std = {std}")
#     print("\n")
#
#     group_dfs.append(group_df)

# regVsClass()
klogtVsDumt()
# weight_decay2()
# chance_for_forbedring()

# nedtaktLilleVsStort()
# kaos()

# group_df = viz0.get_group_df('eksp2-md17_2', remove_nan=False)
# viz0.sanity_check_group_df(group_df)
# if create_cache:
#     df = pd.concat(group_dfs)
#     cache_folder = os.path.join("reports", "manuelt_arkiv")
#     os.makedirs(cache_folder, exist_ok=True)
#     path = os.path.join(cache_folder, f"{prefix}.pickle")
#     cache = {'df': df}
#     with open(path, 'wb') as f:
#         pickle.dump(cache, f)

