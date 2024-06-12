import random
import argparse

# Opret parseren
parser = argparse.ArgumentParser(description="Træk tilfældige heltal mellem 0 og 1001 uden tilbagelægning")

# Tilføj argumentet for n
parser.add_argument('n', type=int, help="Antal tilfældige heltal der skal trækkes")

# Parse kommandolinjeargumenterne
args = parser.parse_args()

n = args.n

# Træk n tilfældige heltal mellem 0 og 1001 uden tilbagelægning
random_numbers = random.sample(range(0, 1002), n)

# Udskriv de tilfældige tal
print("De tilfældige tal er:", random_numbers)