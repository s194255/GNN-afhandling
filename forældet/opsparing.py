import numpy as np
import copy
import matplotlib.pyplot as plt

# startkapital = 200*10**3
# afkast_år = 1.05
# lønstigning_år = 1.00
# indbetaling_måned = 10*10**3
# n_år = 10

def ops(startkapital):
    afkast_år = 1.05
    lønstigning_år = 1.00
    indbetaling_måned = 5 * 10 ** 3
    n_år = 50

    r_måned = afkast_år ** (1 / 12)
    n_måned = n_år*12
    lønstigning_måned = lønstigning_år**(1/12)
    kapital_liste = []
    kapital = copy.deepcopy(startkapital)
    for i in range(n_måned):
        kapital += indbetaling_måned
        kapital *= r_måned
        indbetaling_måned *= lønstigning_måned
        if i % 12 == 0:
            kapital_liste.append(kapital)
    print(f"kapital = {kapital/10**6:.3f} mio")
    return kapital_liste, kapital


if __name__ == "__main__":
    # afkast_år = 1.05
    # lønstigning_år = 1.00
    # indbetaling_måned = 10 * 10 ** 3
    # n_år = 10
    kapital_liste1, kapital1 = ops(-100*10**3)
    kapital_liste2, kapital2 = ops(100*10**3)
    # plt.plot(kapital_liste1)
    # plt.plot(kapital_liste2)
    # plt.show()
    # plt.plot(np.array(kapital_liste2)-np.array(kapital_liste1))
    # plt.show()
    print(f"kapital = {(kapital2-kapital1)/10**6:.3f} mio")