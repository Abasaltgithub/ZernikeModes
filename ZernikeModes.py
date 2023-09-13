import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def zernike_rad(m, n, rho):
    if (n < 0) or (m < 0) or (np.abs(m) > n):
        raise ValueError("Invalid (m, n)")
    if (n - m) % 2:
        return rho * 0.0
    else:
        rad = np.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            c = (-1.0) ** k * comb(n - k, k) * \
                comb(n - 2 * k, (n - m) // 2 - k)
            rad += c * rho ** (n - 2 * k)
        return rad


def zernike(m, n, rho, phi):
    if m > 0:
        return np.sqrt(2) * zernike_rad(m, n, rho) * np.cos(m * phi)
    if m < 0:
        return np.sqrt(2) * zernike_rad(-m, n, rho) * np.sin(-m * phi)
    return zernike_rad(0, n, rho)


# Define a grid on which to calculate the Zernike modes
rho = np.linspace(0, 1, 100)
phi = np.linspace(0, 2 * np.pi, 100)
r, theta = np.meshgrid(rho, phi)
z = np.zeros_like(r)

# Define the Zernike modes to be plotted
modes = [(0, 0), (-1, 1), (1, 1), (-2, 2), (0, 2),
         (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3)]

# Set up the plot
fig, axs = plt.subplots(1, len(modes), figsize=(
    18, 4), subplot_kw=dict(projection='polar'))

# Disable the grid for all subplots
for ax in axs:
    ax.grid(False)

for i, mode in enumerate(modes):
    # Calculate the Zernike mode
    Z = zernike(mode[0], mode[1], r, theta)
    Z = Z.reshape(r.shape)

    axs[i].pcolormesh(theta, r, Z, cmap='RdBu')
    axs[i].set_title(f"Z{mode[0]}{mode[1]}")

# Adjust the spacing between subplots
fig.subplots_adjust(wspace=0.5)

for ax in axs:
    ax.set_xticklabels([])

# Remove color bar and set title
for ax, title in zip(axs, modes):
    ax.set_yticklabels([])
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_title(f"Z{title}")

plt.show()  # Display the plots
