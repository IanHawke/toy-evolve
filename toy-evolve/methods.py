import numpy
from flux_functions import lax_friedrichs, upwind
from slope_limiting import minmod, slope_limited, constant, vanleer
from weno import weno
from functools import partial

def rea_method(reconstruction, flux_solver):
    def rea_solver(q, simulation):
        q_m, q_p = reconstruction(q, simulation)
        flux = flux_solver(q_m, q_p, simulation)
        rhs = numpy.zeros_like(flux)
        rhs[:,1:-1] = 1/simulation.dx * (flux[:,1:-1] - flux[:,2:])
        return rhs
    return rea_solver
    
def rea_method_source(reconstruction, flux_solver, source_term):
    def rea_solver(q, simulation):
        q_m, q_p = reconstruction(q, simulation)
        flux = flux_solver(q_m, q_p, simulation)
        rhs = source_term(q)
        rhs[:,1:-1] += 1/simulation.dx * (flux[:,1:-1] - flux[:,2:])
        return rhs
    return rea_solver
    
constant_lf = rea_method(constant, lax_friedrichs)
constant_upwind = rea_method(constant, upwind)

slope_minmod = partial(slope_limited, limiter=minmod)
minmod_lf = rea_method(slope_minmod, lax_friedrichs)
minmod_upwind = rea_method(slope_minmod, upwind)
slope_vanleer = partial(slope_limited, limiter=vanleer)
vanleer_lf = rea_method(slope_vanleer, lax_friedrichs)
vanleer_upwind = rea_method(slope_vanleer, upwind)

weno3 = partial(weno, order=2)
weno3_lf = rea_method(weno3, lax_friedrichs)
weno3_upwind = rea_method(weno3, upwind)
weno5 = partial(weno, order=3)
weno5_lf = rea_method(weno5, lax_friedrichs)
weno5_upwind = rea_method(weno5, upwind)
