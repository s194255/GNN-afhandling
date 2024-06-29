import numpy as np
import matplotlib.pyplot as plt
import os

n = 100

rod = "reports/figures/teori"
os.makedirs(rod, exist_ok=True)


img = np.random.normal(size=(32, 32))
fig, ax = plt.subplots(1, 1, figsize=(4,4))

# cmaps = ['viridis', 'plasma', 'inferno','magma']


ax.imshow(img, cmap='inferno')
ax.axis('off')
plt.savefig(os.path.join(rod, "støj.svg"))
plt.close()