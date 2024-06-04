import shutil
import os
import matplotlib.pyplot as plt
import src.visualization.farver as farver
from src.visualization import viz0
from tqdm import tqdm
import numpy as np
from scipy.stats import norm

rod = 'reports/figures/Eksperimenter/2/fordeling_af_MAE'
temperaturer = ['frossen', 'optøet']

if os.path.exists(rod):
    shutil.rmtree(rod)



groups, runs = viz0.get_groups_runs('eksp4')
for group in tqdm(groups):
    if group not in ['eksp4_1']:
        continue
    runs_in_group, fortræningsudgaver, temperaturer_lp, seeds, rygrad_runids = viz0.get_loops_params(group, runs)

    assert len(fortræningsudgaver) == 1
    temperaturer_lp = [temperatur for temperatur in temperaturer if temperatur in temperaturer_lp]
    fortræningsudgave = list(fortræningsudgaver)[0]
    kørsel_path = os.path.join(rod, group)
    os.makedirs(kørsel_path)

    plt.figure(figsize=(10, 6))
    for i, temperatur in enumerate(temperaturer_lp):
        runs_filtered = list(filter(lambda w: viz0.main_filter(w, temperatur, fortræningsudgave, None), runs_in_group))
        col = 'test_loss_mean'
        df = viz0.get_df(runs_filtered)
        # df[col] = np.log(df[col])
        prefix = f'{fortræningsudgave}'

        mean = df[col].mean()
        std_dev = df[col].std()
        print(temperatur, mean, std_dev)

        x = np.linspace(df[col].min(), df[col].max(), 100)
        fit = norm.pdf(x, mean, std_dev)
        # fit = lognorm.pdf(x, mean, std_dev)


        bins = 10
        plt.hist(df[col], bins=bins, density=True, alpha=0.9, color=farver.farver[i], edgecolor='black', label=temperatur)  # Histogram
        # plt.scatter(x=df[col], y=[0.0005]*len(df), color=farver.blue, marker='x', s=200)

        # plt.plot(x, fit, farver.farver[i], label=temperatur,
        #          linewidth=5)  # Fittet normalfordeling
    plt.xlabel("MAE", fontsize=18)
    # plt.ylabel('Frekvens', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=14)
    plt.title('Fordeling af MAE', fontsize=22)
    # plt.xscale("log")
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.gca().yaxis.set_ticks([])
    plt.savefig(os.path.join(kørsel_path, "fordeling_af_MAE.jpg"))
    plt.savefig(os.path.join(kørsel_path, "fordeling_af_MAE.pdf"))
    plt.close()