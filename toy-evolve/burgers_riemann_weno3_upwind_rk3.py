# Burgers test evolution: just one

from models import burgers
from bcs import outflow
from simulation import simulation
from methods import weno3_upwind
from rk import rk3
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 100
interval = grid([-1, 1], Npoints, Ngz)
model = burgers.burgers(initial_data = burgers.initial_riemann(1, 0))
sim = simulation(model, interval, weno3_upwind, rk3, outflow, cfl=0.1)
sim.evolve(0.5)

sim.plot_scalar_vs_initial()
pyplot.show()
