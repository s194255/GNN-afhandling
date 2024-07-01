import shutil
import os
import matplotlib.pyplot as plt
import src.visualization.farver as far
from src.visualization import viz0
from tqdm import tqdm
import numpy as np
import seaborn as sns
from scipy import stats


def violinplots(dfs):
    col = 'test_loss_mean'
    # df = {viz0.FORT_LABELLER[fort]: np.log(dfs[fort][col]) for fort in forv_fortræningsudgaver}
    df = {viz0.FORT_LABELLER[fort]: dfs[fort][col] for fort in forv_fortræningsudgaver}
    palette = [farveopslag[fort] for fort in forv_fortræningsudgaver]
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, palette=palette)

    plt.xticks(fontsize=18)  # Endre fontstørrelsen til ønsket verdi
    plt.yticks(fontsize=18)
    plt.ylabel('MAE', fontsize=18)


    # plt.title('Violinfigur')
    plt.savefig(os.path.join(kørsel_path, "stat_sign_violin.jpg"))
    plt.savefig(os.path.join(kørsel_path, "stat_sign_violin.pdf"))


def qqplot(dfs):
    forter = list(dfs.keys())
    col = 'test_loss_mean'

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for i, fort in enumerate(forter):
        df = dfs[fort]
        data = df[col].dropna()
        # data = np.log(data)

        dot_color = farveopslag[fort]
        line_color = far.black

        # Beregn teoretiske kvantiler og sample kvantiler
        (quantiles, values), (slope, intercept, r) = stats.probplot(data, dist="norm")
        f = lambda x: slope * x + intercept

        # Plott kvantilene
        axes[i].scatter(quantiles, values, color=dot_color, edgecolor=dot_color)

        # Plott linjen
        x = np.linspace(quantiles.min(), quantiles.max(), num=1000)
        y = f(x)
        axes[i].plot(x, y, color=line_color)

        titel = viz0.FORT_LABELLER[fort]
        axes[i].set_title(titel, fontsize=32)
        axes[i].tick_params(axis='both', which='major', labelsize=20)
        axes[i].set_xlabel('Teoretiske kvantiler', fontsize=24)
        axes[i].set_ylabel('Observerede kvantiler', fontsize=24)

    plt.tight_layout()
    plt.savefig(os.path.join(kørsel_path, "qq_plots.jpg"))
    plt.savefig(os.path.join(kørsel_path, "qq_plots.pdf"))
    plt.close()

def welsh_t_test(dfs):
    col = 'test_loss_mean'
    a = dfs['uden'][col]
    b = dfs['3D-EMGP-begge'][col]
    alternative = 'greater'
    equal_var = False
    t, p = stats.ttest_ind(a, b, alternative=alternative, equal_var=equal_var)
    print(f"middelværdi af ingen fortræning = {np.mean(a)}")
    print(f"middelværdi af 3D-EMGP-begge = {np.mean(b)}")
    print(f"sigma af ingen fortræning = {np.std(a)}")
    print(f"sigma af 3D-EMGP-begge = {np.std(b)}")
    print(f"p-værdi = {p}")
    print("\n")




def bootstrap_t_test(a, b, num_bootstrap=10**7):
    def bootstrap(data, num_bootstrap):
        samples = np.random.choice(data, size=(num_bootstrap, len(data)), replace=True)
        # Beregn middelværdierne for hver stikprøve
        means = samples.mean(axis=1)
        return means

    a_bs = bootstrap(a, num_bootstrap)
    b_bs = bootstrap(b, num_bootstrap)

    diffs = a_bs - b_bs
    # plt.hist(diffs)
    # plt.show()
    print(diffs.min())

    p_value = np.mean(diffs < 0)
    observed_diff = np.mean(a) - np.mean(b)

    return observed_diff, p_value


def bootstrap_analysis(dfs):
    col = 'test_loss_mean'
    a = dfs['uden'][col].values
    b = dfs['3D-EMGP-begge'][col].values


    observed_diff, p_value = bootstrap_t_test(a, b)

    print(f"Middelværdi af ingen fortræning = {np.mean(a)}")
    print(f"Middelværdi af 3D-EMGP-begge = {np.mean(b)}")
    print(f"Sigma af ingen fortræning = {np.std(a)}")
    print(f"Sigma af 3D-EMGP-begge = {np.std(b)}")
    print(f"Observeret forskel = {observed_diff}")
    print(f"p-værdi = {p_value}")


rod = 'reports/figures/Eksperimenter/3/statistisk_signifikans'
groups = ['eksp3_0']

temperaturer = ['frossen', 'optøet']

if os.path.exists(rod):
    shutil.rmtree(rod)

farveopslag = {
    '3D-EMGP-lokalt': far.bright_green,
    'uden': far.corporate_red,
    '3D-EMGP-begge': far.navy_blue,
}

forv_fortræningsudgaver = ['uden', '3D-EMGP-begge']

# groups, runs = viz0.get_groups_runs('eksp3')
for group in tqdm(groups):
    group_df = viz0.get_group_df(group)
    fortræningsudgaver, temperaturer_lp, seeds = viz0.get_loop_params_group_df(group_df)

    assert set(fortræningsudgaver) == set(forv_fortræningsudgaver)
    assert set(temperaturer_lp) == {'optøet'}

    temperatur = list(temperaturer_lp)[0]
    kørsel_path = os.path.join(rod, group)
    print(kørsel_path)
    os.makedirs(kørsel_path)

    dfs = {}
    for i, fortræningsudgave in enumerate(fortræningsudgaver):
        # idxs = (group_df['fortræningsudgave'] == fortræningsudgave) & (group_df['temperatur'] == temperatur)
        idxs = (group_df['fortræningsudgave'] == fortræningsudgave)
        idxs = idxs & (group_df['temperatur'] == temperatur)
        df = group_df[idxs]
        dfs[fortræningsudgave] = df

    violinplots(dfs)
    qqplot(dfs)
    welsh_t_test(dfs)
    bootstrap_analysis(dfs)


