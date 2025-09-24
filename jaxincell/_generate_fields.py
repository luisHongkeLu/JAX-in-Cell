import numpy as np

G = 40          # must match TOML
L = 1.0          # must match TOML
Bmin = 0.5       # Tesla at center
R = 3.0          # mirror ratio => Bmax = R * Bmin at the ends

x = np.linspace(0.0, L, G)
Bx = Bmin * (1.0 + (R - 1.0) * np.cos(np.pi * x / L)**2)  # min center, max ends

B_ext = np.zeros((G, 3))
B_ext[:, 0] = Bx  # only z component

np.savez("/Users/luis_lu/Desktop/UW-Madison/UWPlasma/aaron/JAX-in-Cell/external_fields/external_B_mirror.npz", B_ext=B_ext, x=x)
print("Wrote external_mirror_B.npz", B_ext.shape)
