"""
Axisymmetric Wave Propagation Example
======================================

Port of k-Wave's ``example_ivp_axisymmetric_simulation.m``.

This example simulates the propagation of an initial pressure distribution
in a heterogeneous medium using the axisymmetric coordinate system. The
2D grid represents the axial (x) and radial (y) directions, with rotational
symmetry about the x-axis.

The medium is divided into two half-spaces with different sound speed and
density, and the initial pressure is a half-disc (which corresponds to a
ball in the full 3D rotated domain).
"""

from functools import partial

import numpy as np
from jax import jit
from jax import numpy as jnp
from matplotlib import pyplot as plt

from jwave.acoustics.axisymmetric import (
    AxialSymmetrySettings,
    simulate_wave_propagation_as,
)
from jwave.geometry import Domain, Medium, TimeAxis


def make_disc(Nx, Ny, cx, cy, radius):
    """Create a binary disc mask."""
    x, y = np.mgrid[0:Nx, 0:Ny]
    return ((x - cx)**2 + (y - cy)**2 < radius**2).astype(float)


def main():
    # =====================================================================
    # SIMULATION SETUP
    # =====================================================================

    # Grid parameters
    Nx = 128    # axial (x) grid points
    Ny = 64     # radial (y) grid points
    dx = 0.1e-3  # axial spacing [m]
    dy = 0.1e-3  # radial spacing [m]

    domain = Domain((Nx, Ny), (dx, dy))

    # Heterogeneous medium — two half-spaces
    sound_speed = np.ones((Nx, Ny)) * 1500.0   # [m/s]
    sound_speed[Nx // 2:, :] = 1800.0

    density = np.ones((Nx, Ny)) * 1000.0   # [kg/m^3]
    density[Nx // 2:, :] = 1200.0

    medium = Medium(
        domain=domain,
        sound_speed=sound_speed,
        density=density,
        pml_size=20,
    )

    # Time axis
    time_axis = TimeAxis.from_medium(medium, cfl=0.3)

    # Initial pressure: half-disc (= ball in 3D when rotated)
    # Generate on a doubled grid, then take the right half
    p0_full = 10.0 * make_disc(Nx, 2 * Ny, Nx // 4 + 8, Ny, 5)
    p0 = p0_full[:, Ny:]  # keep y >= 0

    # =====================================================================
    # RUN SIMULATION
    # =====================================================================

    settings = AxialSymmetrySettings(smooth_initial=True)

    @partial(jit, backend="cpu")
    def run():
        return simulate_wave_propagation_as(
            medium,
            time_axis,
            p0=p0,
            settings=settings,
        )

    print(f"Grid: {Nx} x {Ny}, dt = {time_axis.dt*1e9:.2f} ns, "
          f"Nt = {int(time_axis.Nt)}, t_end = {time_axis.t_end*1e6:.2f} µs")
    print("Running axisymmetric simulation...")
    result = run()
    print("Done!")

    # Result is the full pressure field at each timestep
    p_final = np.asarray(result[-1])
    print(f"Final pressure — min: {p_final.min():.4f}, max: {p_final.max():.4f}")

    # =====================================================================
    # VISUALISATION
    # =====================================================================

    pml = int(medium.pml_size)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Initial pressure
    ax = axes[0]
    ax.set_title("Initial Pressure")
    im = ax.imshow(p0[pml:-pml, :-pml].T, origin="lower",
                   aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("x (axial)")
    ax.set_ylabel("y (radial)")
    plt.colorbar(im, ax=ax, label="Pa")

    # Sound speed
    ax = axes[1]
    ax.set_title("Sound Speed")
    im = ax.imshow(sound_speed[pml:-pml, :-pml].T, origin="lower",
                   aspect="auto", cmap="viridis")
    ax.set_xlabel("x (axial)")
    ax.set_ylabel("y (radial)")
    plt.colorbar(im, ax=ax, label="m/s")

    # Final pressure
    ax = axes[2]
    ax.set_title(f"Pressure at t = {time_axis.t_end*1e6:.1f} µs")
    vmax = np.max(np.abs(p_final[pml:-pml, :-pml])) * 0.8
    im = ax.imshow(p_final[pml:-pml, :-pml].T, origin="lower",
                   aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("x (axial)")
    ax.set_ylabel("y (radial)")
    plt.colorbar(im, ax=ax, label="Pa")

    plt.tight_layout()
    plt.savefig("axisymmetric_result.png", dpi=150, bbox_inches="tight")
    print("Saved plot to axisymmetric_result.png")
    plt.show()


if __name__ == "__main__":
    main()
