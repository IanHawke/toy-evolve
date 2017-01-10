# Burgers test evolution: just one

from models import burgers
from bcs import periodic
from simulation import simulation
from methods import weno3_lf
from rk import rk3
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 400
interval = grid([-1, 1], Npoints, Ngz)
model = burgers.burgers(initial_data = burgers.initial_square())
sim = simulation(model, interval, weno3_lf, rk3, periodic)
sim.evolve(0.5)

sim.plot_scalar_vs_initial()
pyplot.show()
    