# Advection test evolution: convergence test

from models import advection
from bcs import periodic
from simulation import simulation
from methods import weno3_upwind
from rk import rk3
from grid import grid

Ngz = 3
Npoints = 40
interval = grid([-0.5, 0.5], Npoints, Ngz)
model = advection.advection(v=1, 
                            initial_data = advection.initial_sine(period=1))
sim = simulation(model, interval, weno3_upwind, rk3, periodic)
sim.evolve(1.0)

sim.plot_scalar_vs_initial()