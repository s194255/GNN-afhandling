import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("reports/figures/res.pickle", "rb") as f:
    data = pickle.load(f)

x = np.linspace(0.0025, 0.8, num=len(data['med']))

colors = ['r', 'b']



plt.figure()

for idx, (key, tensor) in enumerate(data.items()):
    plt.plot(x, tensor, color=colors[idx], label=key)

plt.xlabel("andel træningdata til downstream")
plt.yscale('log')
plt.legend()
print("nu skal jeg til at vise")
plt.savefig("reports/figures/eksp2.jpg")



