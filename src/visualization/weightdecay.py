import pandas as pd

from src.visualization import viz0
import numpy as np
import os
import pickle

def regVsClass():
    runid_to_loss = {
        '7cajugge': 'class',
        'x96gc9d0': 'reg'
    }
    group_df = viz0.get_group_df('RegVsClass_1')
    a = 2
    for k, v in runid_to_loss.items():
        ixs = group_df['rygrad runid'] == k
        mean = group_df[ixs]['test_loss_mean'].mean()
        std = group_df[ixs]['test_loss_mean'].std()
        print(f"loss = {v}")
        print(f"mean = {mean}")
        print(f"std = {std}")
        print("\n")


# kørselid = [0, 1, 2, 3, 4, 5, 6, 7]
# groups = [f'weightdecay_{k}' for k in kørselid]

create_cache = False

# prefix = 'støjniveau'
# kørselid = [0, 1, 2]
prefix = 'weightdecay'
kørselid = [0, 1, 2, 3, 4, 5, 6, 7]


groups = [f'{prefix}_{k}' for k in kørselid]
group_dfs = []
for k, group in zip(kørselid, groups):
    group_df = viz0.get_group_df(group)
    weightdecay = group_df['weight_decay'].unique()
    print(k)
    print(np.log10(weightdecay))
    mean = group_df['test_loss_mean'].mean()
    std = group_df['test_loss_mean'].std()
    print(f"mean = {mean}")
    print(f"std = {std}")
    print("\n")

    group_dfs.append(group_df)

regVsClass()

if create_cache:
    df = pd.concat(group_dfs)
    cache_folder = os.path.join("reports", "manuelt_arkiv")
    os.makedirs(cache_folder, exist_ok=True)
    path = os.path.join(cache_folder, f"{prefix}.pickle")
    cache = {'df': df}
    with open(path, 'wb') as f:
        pickle.dump(cache, f)

