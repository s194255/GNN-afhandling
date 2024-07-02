import random
import argparse
from src.visualization.viz0 import get_group_df

# Opret parseren
parser = argparse.ArgumentParser(description="Træk tilfældige heltal mellem 0 og 1001 uden tilbagelægning")

# Tilføj argumentet for n
parser.add_argument('--n', type=int, help="Antal tilfældige heltal der skal trækkes")

parser.add_argument('--group', type=str, help="lol")

# Parse kommandolinjeargumenterne
args = parser.parse_args()

n = args.n

pool = range(0, 1002)

if args.group != 'null':
    group_df = get_group_df(args.group)
    seeds = group_df['seed'].unique()
    print(f"Du har brugt disse {len(seeds)} frø: {seeds}. Dem springer jeg over")
    pool = list(set(pool) - set(seeds))

# Træk n tilfældige heltal mellem 0 og 1001 uden tilbagelægning
random_numbers = random.sample(range(0, 1002), n)

# Udskriv de tilfældige tal
print("De tilfældige tal er:", random_numbers)