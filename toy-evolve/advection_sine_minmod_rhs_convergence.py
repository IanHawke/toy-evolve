# Advection test RHS: convergence test

from models import advection
from bcs import periodic
from simulation import simulation
from methods import minmod_upwind, minmod_lf, constant_lf
from rk import rk2, rk3
from grid import grid
import numpy
from matplotlib import pyplot

Ngz = 2
Npoints_all = 40 * 2**numpy.arange(6)
dx_all = 1 / Npoints_all

t_end = 0.05

for i, Npoints in enumerate(Npoints_all):
    print(Npoints)
    interval = grid([-0.5, 0.5], Npoints, Ngz)
    model = advection.advection(v=1, 
                                initial_data = advection.initial_sine(period=1))
    sim = simulation(model, interval, minmod_lf, rk3, periodic)
    sim.evolve(t_end)
    rhs = minmod_upwind(sim.q0, sim)
    x = sim.coordinates[Ngz:-2*Ngz]
    xbar = x - t_end
    xbar[numpy.abs(xbar)>0.5] -= numpy.sign(xbar[numpy.abs(xbar)>0.5])
    pyplot.plot(x, Npoints**3*(sim.q[0,Ngz:-2*Ngz]-numpy.sin(2*numpy.pi*xbar)))
pyplot.ylim(-1000,1000)
pyplot.show()
