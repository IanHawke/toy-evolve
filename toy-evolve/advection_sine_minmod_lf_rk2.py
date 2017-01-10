# Advection test evolution

from models import advection
from bcs import periodic
from simulation import simulation
from methods import minmod_lf
from rk import rk2
from grid import grid

Npoints = 20
Ngz = 2
interval = grid([-0.5, 0.5], Npoints, Ngz)
model = advection.advection(v=1, 
                            initial_data = advection.initial_sine(period=1))
model = advection.advection(v=1, 
                            initial_data = advection.initial_square())
sim = simulation(model, interval, minmod_lf, rk2, periodic)
sim.evolve(0.5)
