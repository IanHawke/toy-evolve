# Relaxation system: relaxes to Burgers as tau->0
#
# Need to work out why upwind RP isn't working
# Need to work out why WENO isn't working (characteristic reconstruction?)
#

import numpy
from models import relaxation_burgers
from bcs import outflow
from simulation import simulation
from methods import weno3_upwind, vanleer_lf, vanleer_upwind
from rk import imex222, rk3
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 200
tau = 1.0
tau2 = 0.01
L = 1
interval = grid([-L, L], Npoints, Ngz)
qL = numpy.array([1.0, 0.5])
qR = numpy.array([0.0, 0.0])
model = relaxation_burgers.relaxation_burgers(initial_data = relaxation_burgers.initial_riemann(qL, qR))
source  = relaxation_burgers.relaxation_source(tau)
source2 = relaxation_burgers.relaxation_source(tau2)

#sim_no_source = simulation(model, interval, vanleer_lf, rk3, outflow, cfl=0.1)
#sim_no_source.evolve(0.6)
#sim_no_source.plot_system()
#pyplot.show()

sim = simulation(model, interval, vanleer_lf, imex222(source), outflow)
sim.evolve(0.6)
sim.plot_system()
pyplot.show()

sim2 = simulation(model, interval, vanleer_lf, imex222(source2), outflow)
sim2.evolve(0.6)
sim2.plot_system()
pyplot.show()
