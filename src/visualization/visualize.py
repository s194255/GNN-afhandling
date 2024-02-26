import matplotlib.pyplot as plt
import pickle

print("nu læser jeg")
with open("reports/figures/res.pickle", "rb") as f:
    data = pickle.load(f)
print("nu er jeg færdig med at læse")

colors = ['r', 'b']

plt.figure()

for idx, (key, tensor) in enumerate(data.items()):
    plt.plot(tensor, color=colors[idx], label=key)

plt.legend()
print("nu skal jeg til at vise")
plt.show()



