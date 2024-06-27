
from src.visualization import viz0
import numpy as np


# kørselid = [0, 1, 2, 3, 4, 5, 6, 7]
# groups = [f'weightdecay_{k}' for k in kørselid]

kørselid = [0, 1, 2]
groups = [f'støjniveau_{k}' for k in kørselid]

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
