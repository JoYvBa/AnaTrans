import numpy as np
import equations_ADE as ade
import ADE_eq as ade2
import matplotlib.pyplot as plt

x = np.linspace(0,10,100)
t = np.linspace(0,20,100)

Ca = ade2.cxt_1D(x, t)
Ca.ade(advection = False, dispersion = True)
print(Ca.cxt)
plt.plot(x,Ca.cxt[-1,:])
plt.show()

# C1 = ade.Concentration(x,t)
# C1.ade(bc='pulse', dispersion = True)

# # C2 = ade.Concentration(x,t)
# # C2.advection()
# plt.plot(x, C1.cxt[:,-1])
# # plt.plot(x, C2.cxt[:,-1], '--')
# plt.show()
