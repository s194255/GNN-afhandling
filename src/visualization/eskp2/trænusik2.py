import shutil
import os
import matplotlib.pyplot as plt
from src.visualization.farver import farver
from tqdm import tqdm
import viz0
import numpy as np

TITLER = {'frossen': "Frossen rygrad",
          'optøet': "Optøet rygrad"}

LABELLER = {'uden': 'Ingen fortræning',
            'Selvvejledt': '3D-EMGP',
            'SelvvejledtQM9': 'QM9 fortræning'}

FIGNAVN = 'trænusik2'
ROOT = os.path.join('reports/figures/Eksperimenter/2', FIGNAVN)


def plot(df):
    # Opsætning for søjlerne
    x_values = df['eftertræningsmængde'].unique()
    x_values.sort()
    fortræningsudgaver = df['fortræningsudgave'].unique()
    num_models = len(fortræningsudgaver)


    bar_width = 0.2
    x = np.arange(len(x_values))

    # Opret figuren og akserne
    fig, ax = plt.subplots()

    # Plot søjlerne og prikkerne
    for i in range(num_models):
        fortræningsudgave = fortræningsudgaver[i]
        målinger = df[df['fortræningsudgave'] == fortræningsudgave][['eftertræningsmængde', 'test_loss_mean']]
        søjlehøjde = målinger.groupby('eftertræningsmængde').mean().reset_index()['test_loss_mean']
        bars = ax.bar(x + (i - num_models / 2) * bar_width, søjlehøjde, bar_width, color=farver[i], alpha=0.5)
        for j in range(len(x_values)):
            prikker = målinger[målinger['eftertræningsmængde'] == x_values[j]]['test_loss_mean']
            n2 = len(prikker)
            label = LABELLER[fortræningsudgave] if j==0 else None
            ax.scatter([x[j] + (i - num_models / 2) * bar_width] * n2, prikker,
                       color=farver[i], label=label, marker='x')

    # Tilpasning af akserne og labels
    ax.set_xlabel('Datamængde', fontsize=16)
    ax.set_ylabel('MAE', fontsize=16)
    ax.set_title(f'{temperatur} rygrad', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values)
    ax.legend(fontsize=18)

    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.jpg"))
    plt.savefig(os.path.join(kørsel_path, f"{temperatur}_{FIGNAVN}.pdf"))

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)

groups, runs = viz0.get_groups_runs('eksp2')
for group in tqdm(groups):
    # if group not in ['eksp2_48']:
    #     continue
    runs_in_group, fortræningsudgaver, temperaturer, seeds, rygrad_runids = viz0.get_loops_params(group, runs)
    eftertræningsmængder = viz0.get_eftertræningsmængder(group, runs)
    assert len(temperaturer) == 1
    temperatur = list(temperaturer)[0]
    kørsel_path = os.path.join(ROOT, group)
    os.makedirs(kørsel_path)

    runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave=None, seed=None), runs_in_group))
    df = viz0.get_df(runs_filtered)
    # df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='any')

    plot(df)
