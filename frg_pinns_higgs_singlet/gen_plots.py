import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("plots_pinn", exist_ok=True)

import show_UV1D
show_UV1D.plot_u(sigma_fixed=0.0)
plt.gcf().savefig("plots_pinn/u_rho_slices.png", dpi=150)
plt.close("all")
print("saved plots_pinn/u_rho_slices.png")

import show_1D_sigma
show_1D_sigma.plot_u(rho_fixed=0.0)
plt.gcf().savefig("plots_pinn/u_sigma_slices.png", dpi=150)
plt.close("all")
print("saved plots_pinn/u_sigma_slices.png")
