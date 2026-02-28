# This file is part of j-Wave.
#
# j-Wave is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# j-Wave is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with j-Wave. If not, see <https://www.gnu.org/licenses/>.

"""
Axisymmetric wave propagation simulation.

This module implements the time-domain simulation of acoustic wave propagation
in an axisymmetric (cylindrical) coordinate system, equivalent to k-Wave's
``kspaceFirstOrderAS``. In this coordinate system:

- The x-dimension corresponds to the **axial** direction
- The y-dimension corresponds to the **radial** direction
- The coordinate system is rotationally symmetric about the x-axis

The key physics difference from a standard 2D simulation is the extra ``u_r/r``
term in the divergence operator arising from cylindrical coordinates:

.. math::

    \\nabla \\cdot \\mathbf{u} = \\frac{\\partial u_x}{\\partial x}
    + \\frac{\\partial u_r}{\\partial r} + \\frac{u_r}{r}

The implementation uses the WSWA-FFT approach: fields are mirrored in the
radial direction with appropriate symmetries and then processed using standard
FFTs on the expanded grid.
"""

from typing import Callable

import equinox as eqx
import numpy as np
from jax import checkpoint as jax_checkpoint
from jax import numpy as jnp
from jax.lax import scan
from jaxdf.mods import Module

from jwave.geometry import Medium, TimeAxis
from jwave.logger import logger
from jwave.signal_processing import smooth


class AxialSymmetrySettings(Module):
    """Settings for axisymmetric wave propagation simulation.

    !!! example
    ```python
    >>> settings = AxialSymmetrySettings()
    >>> print(settings.checkpoint)
    True
    ```
    """

    c_ref: Callable = eqx.field(static=True)
    checkpoint: bool = eqx.field(static=True)
    smooth_initial: bool = eqx.field(static=True)

    def __init__(
        self,
        c_ref: Callable = lambda m: m.max_sound_speed,
        checkpoint: bool = True,
        smooth_initial: bool = True,
    ):
        """
        Args:
            c_ref: Callable that takes the ``medium`` and returns the
                reference sound speed for the k-space operator.
            checkpoint: Whether to use JAX checkpointing to save memory
                during backpropagation.
            smooth_initial: Whether to smooth initial pressure field.
        """
        self.c_ref = c_ref
        self.checkpoint = checkpoint
        self.smooth_initial = smooth_initial


# =========================================================================
# WSWA-FFT mirroring utilities
# =========================================================================

def _mirror_wswa(field, Ny):
    """Mirror a 2D field using WSWA (Whole-Sample Whole-sample Antisymmetric)
    symmetry. Used for pressure and axial velocity.

    The original field occupies columns [0:Ny], and the expanded field has
    4*Ny columns following the pattern (k-Wave convention, 0-indexed):
    ``[f, 0, -f[N-2::-1], -f, 0, f[N-2::-1]]``

    In other words the antisymmetric mirror uses ``f[:-1]`` (excludes the
    last element) **not** ``f[1:]`` (excludes the first element).  Using
    ``f[1:]`` is an off-by-one error that shifts the mirror tile by one
    grid cell and breaks the spectral anti-symmetry.

    Args:
        field: Array of shape ``(Nx, Ny)``.
        Ny: Number of radial grid points (original).

    Returns:
        Expanded array of shape ``(Nx, 4*Ny)``.
    """
    Nx = field.shape[0]
    expanded = jnp.zeros((Nx, 4 * Ny), dtype=field.dtype)

    # [0 : Ny] -> original
    expanded = expanded.at[:, :Ny].set(field)
    # [Ny] -> 0  (whole-sample antisymmetric boundary; stays zero)
    # [Ny+1 : 2*Ny] -> -flip(f[:, :-1])  i.e. -[f[N-2], ..., f[0]]
    expanded = expanded.at[:, Ny + 1:2 * Ny].set(-jnp.flip(field[:, :-1], axis=1))
    # [2*Ny : 3*Ny] -> -f
    expanded = expanded.at[:, 2 * Ny:3 * Ny].set(-field)
    # [3*Ny] -> 0  (stays zero)
    # [3*Ny+1 : 4*Ny] -> flip(f[:, :-1])  i.e. [f[N-2], ..., f[0]]
    expanded = expanded.at[:, 3 * Ny + 1:4 * Ny].set(jnp.flip(field[:, :-1], axis=1))

    return expanded


def _mirror_hahs(field, Ny):
    """Mirror a 2D field using HAHS (Half-sample Antisymmetric Half-sample
    Symmetric) symmetry. Used for radial velocity.

    Args:
        field: Array of shape ``(Nx, Ny)``.
        Ny: Number of radial grid points (original).

    Returns:
        Expanded array of shape ``(Nx, 4*Ny)``.
    """
    Nx = field.shape[0]
    expanded = jnp.zeros((Nx, 4 * Ny), dtype=field.dtype)

    # [0 : Ny] -> original
    expanded = expanded.at[:, :Ny].set(field)
    # [Ny : 2*Ny-1] -> fliplr(field)
    expanded = expanded.at[:, Ny:2 * Ny].set(jnp.flip(field, axis=1))
    # [2*Ny : 3*Ny-1] -> -field
    expanded = expanded.at[:, 2 * Ny:3 * Ny].set(-field)
    # [3*Ny : 4*Ny-1] -> -fliplr(field)
    expanded = expanded.at[:, 3 * Ny:4 * Ny].set(-jnp.flip(field, axis=1))

    return expanded


def _mirror_hsha(field_over_r, Ny):
    """Mirror ``u_r / r`` using HSHA (Half-sample Symmetric Half-sample
    Antisymmetric) symmetry.

    Args:
        field_over_r: Array of shape ``(Nx, Ny)`` containing ``u_r / r``.
        Ny: Number of radial grid points (original).

    Returns:
        Expanded array of shape ``(Nx, 4*Ny)``.
    """
    Nx = field_over_r.shape[0]
    expanded = jnp.zeros((Nx, 4 * Ny), dtype=field_over_r.dtype)

    # [0 : Ny] -> original
    expanded = expanded.at[:, :Ny].set(field_over_r)
    # [Ny : 2*Ny-1] -> -fliplr(field_over_r)
    expanded = expanded.at[:, Ny:2 * Ny].set(-jnp.flip(field_over_r, axis=1))
    # [2*Ny : 3*Ny-1] -> -field_over_r
    expanded = expanded.at[:, 2 * Ny:3 * Ny].set(-field_over_r)
    # [3*Ny : 4*Ny-1] -> fliplr(field_over_r)
    expanded = expanded.at[:, 3 * Ny:4 * Ny].set(jnp.flip(field_over_r, axis=1))

    return expanded


# =========================================================================
# k-space operators for the axisymmetric code
# =========================================================================

def _build_axisymmetric_operators(Nx, dx, Ny, dy, dt, c_ref):
    """Build the derivative and k-space operators for the axisymmetric code.

    The expanded grid is 4*Ny in the radial direction.

    Args:
        Nx: Number of axial grid points.
        dx: Axial grid spacing.
        Ny: Number of radial grid points.
        dy: Radial grid spacing.
        dt: Time step.
        c_ref: Reference sound speed.

    Returns:
        Dictionary of operators.
    """
    Ny_exp = 4 * Ny

    # Axial (x) wavenumbers — same as standard 2D
    kx_vec = jnp.fft.fftfreq(Nx, dx) * 2 * jnp.pi

    # Radial (y) wavenumbers — on the expanded grid
    ky_vec = jnp.fft.fftfreq(Ny_exp, dy) * 2 * jnp.pi

    # Derivative operators for x (column vectors)
    ddx_k_shift_pos = (1j * kx_vec * jnp.exp(1j * kx_vec * dx / 2))[:, None]
    ddx_k_shift_neg = (1j * kx_vec * jnp.exp(-1j * kx_vec * dx / 2))[:, None]

    # Derivative operator for y (row vectors)
    ddy_k = (1j * ky_vec)[None, :]

    # Shift operators for y
    y_shift_pos = jnp.exp(1j * ky_vec * dy / 2)[None, :]
    y_shift_neg = jnp.exp(-1j * ky_vec * dy / 2)[None, :]

    # k-space correction operator (sinc)
    kx_grid = kx_vec[:, None]
    ky_grid = ky_vec[None, :]
    k_magnitude = jnp.sqrt(kx_grid**2 + ky_grid**2)
    kappa = jnp.sinc(c_ref * k_magnitude * dt / (2 * jnp.pi))

    # Radial distance vectors (non-staggered and staggered)
    # In the axisymmetric code, y starts at 0 (axis of symmetry)
    y_vec = jnp.arange(Ny) * dy
    y_vec_sg = y_vec + dy / 2  # staggered grid (shifted by dy/2)

    return {
        "ddx_k_shift_pos": ddx_k_shift_pos,
        "ddx_k_shift_neg": ddx_k_shift_neg,
        "ddy_k": ddy_k,
        "y_shift_pos": y_shift_pos,
        "y_shift_neg": y_shift_neg,
        "kappa": kappa,
        "y_vec_sg": y_vec_sg,
    }


# =========================================================================
# PML for the axisymmetric case
# =========================================================================

def _axisymmetric_pml(Nx, Ny, dx, dy, dt, c_ref, pml_size,
                      alpha_max=2.0, exponent=4.0, coord_shift=0.0):
    """Build PML absorption arrays for the axisymmetric case.

    In the axial (x) direction, PML is applied on both sides.
    In the radial (y) direction, PML is only applied on the outer edge
    (not at y=0 which is the axis of symmetry).

    Args:
        Nx, Ny: Grid dimensions.
        dx, dy: Grid spacings.
        dt: Time step.
        c_ref: Reference sound speed.
        pml_size: PML thickness in grid points.
        alpha_max: PML absorption coefficient.
        exponent: PML profile exponent.
        coord_shift: Grid stagger offset (0 for pressure, 0.5 for velocity).

    Returns:
        Tuple of (pml_x, pml_y) arrays with shapes ``(Nx, 1)`` and ``(1, Ny)``.
    """
    if pml_size == 0:
        return jnp.ones((Nx, 1)), jnp.ones((1, Ny))

    int_pml = int(pml_size)

    # --- Axial (x) PML: both sides ---
    x_right = ((jnp.arange(1, int_pml + 1) + coord_shift) / pml_size) ** exponent
    x_left = ((jnp.arange(int_pml, 0, -1) - coord_shift) / pml_size) ** exponent

    alpha_x_left = jnp.exp(alpha_max * (-1) * x_left * dt * c_ref / 2 / dx)
    alpha_x_right = jnp.exp(alpha_max * (-1) * x_right * dt * c_ref / 2 / dx)

    pml_x = jnp.ones(Nx)
    pml_x = pml_x.at[:int_pml].set(alpha_x_left)
    pml_x = pml_x.at[-int_pml:].set(alpha_x_right)
    pml_x = pml_x[:, None]  # shape (Nx, 1)

    # --- Radial (y) PML: outer edge only ---
    y_right = ((jnp.arange(1, int_pml + 1) + coord_shift) / pml_size) ** exponent
    alpha_y_right = jnp.exp(alpha_max * (-1) * y_right * dt * c_ref / 2 / dy)

    pml_y = jnp.ones(Ny)
    pml_y = pml_y.at[-int_pml:].set(alpha_y_right)
    pml_y = pml_y[None, :]  # shape (1, Ny)

    return pml_x, pml_y


# =========================================================================
# Main simulation function
# =========================================================================

def simulate_wave_propagation_as(
    medium,
    time_axis,
    *,
    settings=None,
    sources=None,
    sensors=None,
    p0=None,
):
    r"""Simulate wave propagation in an axisymmetric coordinate system.

    This is the j-wave equivalent of k-Wave's ``kspaceFirstOrderAS``. The
    simulation operates on a 2D grid where the x-dimension is the axial
    direction and the y-dimension is the radial direction. The coordinate
    system is rotationally symmetric about the x-axis.

    The y=0 boundary (first column) corresponds to the axis of symmetry.
    A point at y=0 maps to a point source in 3D; a point at y>0 maps to
    a ring source in 3D.

    Args:
        medium: A ``Medium`` object. Sound speed and density can be given
            as scalars (homogeneous) or as ``FourierSeries`` / arrays of
            shape ``(Nx, Ny)`` (heterogeneous). The ``domain`` must be 2D
            with shape ``(Nx, Ny)`` where x is axial and y is radial.
        time_axis: ``TimeAxis`` object specifying ``dt`` and ``t_end``.
        settings: ``AxialSymmetrySettings`` controlling simulation options.
            If ``None``, default settings are used.
        sources: Source terms (currently unused, reserved for future use).
        sensors: Callable ``sensors(p, u_x, u_y)`` returning recorded data.
            If ``None``, the full pressure field is returned.
        p0: Initial pressure field as a 2D array of shape ``(Nx, Ny)``.
            This represents the half-plane; the first column (y=0) is on
            the axis of symmetry.

    Returns:
        Sensor recordings at each time step. By default, returns the full
        pressure field at every step.
    """
    if settings is None:
        settings = AxialSymmetrySettings()

    # Domain info
    Nx, Ny = medium.domain.N
    dx, dy = medium.domain.dx
    dt = time_axis.dt

    import jax
    # Reference sound speed
    c_ref = jax.lax.stop_gradient(settings.c_ref(medium))

    # Extract medium properties as arrays
    from jaxdf import Field
    def _to_array(x, shape):
        if isinstance(x, Field):
            return x.on_grid[..., 0]
        elif jnp.isscalar(x) or (hasattr(x, 'shape') and x.shape == ()):
            return jnp.ones(shape) * x
        else:
            return jnp.asarray(x)

    c0 = _to_array(medium.sound_speed, (Nx, Ny))
    rho0 = _to_array(medium.density, (Nx, Ny))

    # Staggered density (interpolated to half-grid positions)
    # sgx = (x + dx/2, y), sgy = (x, y + dy/2)
    rho0_sgx = 0.5 * (rho0 + jnp.roll(rho0, -1, axis=0))
    rho0_sgy = 0.5 * (rho0 + jnp.roll(rho0, -1, axis=1))
    rho0_sgx_inv = 1.0 / rho0_sgx
    rho0_sgy_inv = 1.0 / rho0_sgy

    # Build k-space and derivative operators
    ops = _build_axisymmetric_operators(Nx, dx, Ny, dy, dt, c_ref)
    ddx_k_shift_pos = ops["ddx_k_shift_pos"]
    ddx_k_shift_neg = ops["ddx_k_shift_neg"]
    ddy_k = ops["ddy_k"]
    y_shift_pos = ops["y_shift_pos"]
    y_shift_neg = ops["y_shift_neg"]
    kappa = ops["kappa"]
    y_vec_sg = ops["y_vec_sg"]

    # Build PML
    pml_size = medium.pml_size
    pml_x, pml_y = _axisymmetric_pml(
        Nx, Ny, dx, dy, dt, c_ref, pml_size, coord_shift=0.0
    )
    pml_x_sgx, pml_y_sgy = _axisymmetric_pml(
        Nx, Ny, dx, dy, dt, c_ref, pml_size, coord_shift=0.5
    )

    # Default sensors
    if sensors is None:
        sensors = lambda p, ux, uy: p

    # Initial conditions
    if p0 is None:
        p = jnp.zeros((Nx, Ny))
    else:
        # Ensure p0 is a plain array
        if isinstance(p0, Field):
            p = p0.on_grid[..., 0]
        else:
            p = jnp.asarray(p0)

        if settings.smooth_initial:
            p = smooth(p)

    # Initialize fields
    rhox = p / (2.0 * c0**2)
    rhoy = p / (2.0 * c0**2)
    ux_sgx = jnp.zeros((Nx, Ny))
    uy_sgy = jnp.zeros((Nx, Ny))

    # If we have initial pressure, compute the initial velocity to force
    # u(t=0) = 0 by using u(t=-dt/2) = dt/2 * grad(p) / rho
    if p0 is not None:
        # Compute gradient of initial pressure on expanded grid
        p_exp = _mirror_wswa(p, Ny)
        p_k = kappa * jnp.fft.fft2(p_exp)
        dpdx_sgx_exp = jnp.fft.ifft2(ddx_k_shift_pos * p_k).real
        dpdy_sgy_exp = jnp.fft.ifft2(ddy_k * y_shift_pos * p_k).real

        dpdx_sgx = dpdx_sgx_exp[:, :Ny]
        dpdy_sgy = dpdy_sgy_exp[:, :Ny]

        # u(t = t1 - dt/2) based on u(dt/2) = -u(-dt/2) => u(t1) = 0
        ux_sgx = dt * rho0_sgx_inv * dpdx_sgx / 2
        uy_sgy = dt * rho0_sgy_inv * dpdy_sgy / 2

    # Time stepping
    output_steps = jnp.arange(0, time_axis.Nt, 1)

    # Pack state into a flat list for scan
    state = (p, rhox, rhoy, ux_sgx, uy_sgy)

    def scan_fun(state, n):
        p, rhox, rhoy, ux_sgx, uy_sgy = state

        # === Compute dp/dx and dp/dy ===
        p_exp = _mirror_wswa(p, Ny)
        p_k = kappa * jnp.fft.fft2(p_exp)
        dpdx_sgx = jnp.fft.ifft2(ddx_k_shift_pos * p_k).real[:, :Ny]
        dpdy_sgy = jnp.fft.ifft2(ddy_k * y_shift_pos * p_k).real[:, :Ny]

        # === Update velocity ===
        ux_sgx = pml_x_sgx * (pml_x_sgx * ux_sgx - dt * rho0_sgx_inv * dpdx_sgx)
        uy_sgy = pml_y_sgy * (pml_y_sgy * uy_sgy - dt * rho0_sgy_inv * dpdy_sgy)

        # === Compute du_x/dx and (du_y/dy + u_y/y) ===
        # Mirror ux using WSWA symmetry
        ux_exp = _mirror_wswa(ux_sgx, Ny)

        # Mirror uy using HAHS symmetry
        uy_exp = _mirror_hahs(uy_sgy, Ny)

        # Mirror uy/y using HSHA symmetry
        uy_over_y = uy_sgy / y_vec_sg[None, :]
        uy_on_y_exp = _mirror_hsha(uy_over_y, Ny)

        # Compute dux/dx on expanded grid
        duxdx_exp = jnp.fft.ifft2(
            ddx_k_shift_neg * kappa * jnp.fft.fft2(ux_exp)
        ).real
        duxdx = duxdx_exp[:, :Ny]

        # Compute duy/dy + uy/y on expanded grid
        # The y-derivative of uy plus the uy/y term, with shift applied
        duydy_exp = jnp.fft.ifft2(
            kappa * y_shift_neg * (
                ddy_k * jnp.fft.fft2(uy_exp) + jnp.fft.fft2(uy_on_y_exp)
            )
        ).real
        duydy = duydy_exp[:, :Ny]

        # === Update density (linearised mass conservation) ===
        rhox = pml_x * (pml_x * rhox - dt * rho0 * duxdx)
        rhoy = pml_y * (pml_y * rhoy - dt * rho0 * duydy)

        # === Update pressure (linear adiabatic equation of state) ===
        p = c0**2 * (rhox + rhoy)

        new_state = (p, rhox, rhoy, ux_sgx, uy_sgy)
        return new_state, sensors(p, ux_sgx, uy_sgy)

    if settings.checkpoint:
        scan_fun = jax_checkpoint(scan_fun)

    logger.debug("Starting axisymmetric simulation (WSWA-FFT)")
    _, ys = scan(scan_fun, state, output_steps)

    return ys
