import shutil
import os
# from src.visualization.farver import farver
import src.visualization.farver as far
from tqdm import tqdm
from src.visualization import viz0
import pandas as pd

TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet rygrad"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9 fortræning',
            '3D-EMGP-lokalt': '3D-EMGP kun lokalt',
            '3D-EMGP-globalt': '3D-EMGP kun globalt',
            '3D-EMGP-begge': '3D-EMGP'
            }

FIGNAVN = 'samfattabel'
ROOT = os.path.join('reports/figures/Eksperimenter/2', FIGNAVN)

farver = [far.corporate_red, far.blue, far.navy_blue, far.bright_green, far.orange, far.yellow]
stjerner = viz0.get_stjerner()


if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    if stjerner != None:
        if group not in [f'eksp2_{udvalgt}' for udvalgt in stjerner]:
            continue
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    eftertræningsmængder = viz0.get_eftertræningsmængder(group, runs)
    assert len(temperaturer) == 1
    temperatur = list(temperaturer)[0]
    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)

    runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave=None, seed=None), runs_in_group))
    df = viz0.get_df(runs_filtered)
    out_df = {'datamængde': []}
    out_df = {**out_df, **{fortræningsudgave: [] for fortræningsudgave in fortræningsudgaver}}
    out_df = pd.DataFrame(data=out_df)

    x_values = df['eftertræningsmængde'].unique()
    x_values.sort()

    for x_value in x_values:
        linje = {'datamængde': [x_value]}
        for fortræningsudgave in fortræningsudgaver:
            idxs = (df['fortræningsudgave'] == fortræningsudgave) & (df['eftertræningsmængde'] == x_value)
            målinger = df[idxs][['eftertræningsmængde', 'test_loss_mean']]
            søjlehøjde = målinger['test_loss_mean'].mean()
            linje[fortræningsudgave] = [søjlehøjde]
        out_df = pd.concat([out_df, pd.DataFrame(data=linje)], ignore_index=True)
    print(out_df)
    # out_df = out_df.round(2)
    latex_table = out_df.to_latex(index=False, float_format="%.2f")
    path = os.path.join(kørsel_path, f"{temperatur}_trænusik.tex")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
