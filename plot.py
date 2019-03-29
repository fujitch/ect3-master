# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

a = np.reshape(dataset25[1][1][0][6:], [31, 3])
b = np.reshape(dataset25[1][3][0][6:], [31, 3])
c = np.reshape(dataset25[1][5][0][6:], [31, 3])

plt.plot(range(31), a[:, 0], label = "L=1mm")
plt.plot(range(31), b[:, 0], label = "L=3mm")
plt.plot(range(31), c[:, 0], label = "L=5mm")

plt.xlabel("Distance from flaw(mm)", size = 14)
plt.ylabel("Coil Voltage(V)", size = 14)

plt.legend(fontsize=20)

plt.show()