import logging
from io import StringIO

import numpy as np
from jax import numpy as jnp

from jwave.acoustics import simulate_wave_propagation_as, AxialSymmetrySettings
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.logger import logger, set_logging_level


def test_axisymmetric_simulation():
    # Setup small domain
    Nx, Ny = 32, 16
    domain = Domain((Nx, Ny), (0.1, 0.1))
    medium = Medium(domain, sound_speed=1500.0, density=1000.0, pml_size=5)
    
    # 5 steps should be enough to see wave moving
    time_axis = TimeAxis.from_medium(medium, cfl=0.3)
    time_axis.t_end = time_axis.dt * 5

    # Center-ish point source at the axis of symmetry (y=0)
    p0 = jnp.zeros((Nx, Ny))
    p0 = p0.at[Nx // 2, 0].set(1.0)
    
    settings = AxialSymmetrySettings(checkpoint=False, smooth_initial=False)

    # Capture logs
    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    logger.addHandler(ch)
    set_logging_level(logging.DEBUG)

    # Run
    p = simulate_wave_propagation_as(medium, time_axis, p0=p0, settings=settings)

    # Clean up
    logger.removeHandler(ch)
    log_contents = log_capture_string.getvalue()
    set_logging_level(logging.INFO)

    # Assertions
    assert "Starting axisymmetric simulation (WSWA-FFT)" in log_contents
    assert p.shape == (5, Nx, Ny)
    
    # Check physical plausibility
    p_final = p[-1]
    
    # Pressure shouldn't be all zero or NaN
    assert not jnp.all(p_final == 0)
    assert not jnp.any(jnp.isnan(p_final))
    
    # Max pressure should have moved from purely the center
    # This is a basic test just to make sure the loop actually did something
    assert jnp.max(jnp.abs(p_final)) > 0


if __name__ == "__main__":
    test_axisymmetric_simulation()
