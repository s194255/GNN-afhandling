import random

# Træk 5 tilfældige heltal mellem 0 og 1001 uden tilbagelægning
random_numbers = random.sample(range(0, 1002), 3)

# Udskriv de tilfældige tal
print("De tilfældige tal er:", random_numbers)
