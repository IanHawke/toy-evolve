# Burgers test evolution: just one

import numpy
from models import burgers
from bcs import outflow
from simulation import simulation
from methods import weno3_upwind
from rk import imex433
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 200
tau = 5e-3
beta = 0.8
L = 1
interval = grid([-L, L], Npoints, Ngz)

q_exact = lambda x, t : 1/(1+numpy.exp(-(x-beta*t)/tau))
x_exact = numpy.linspace(-L,L,1000)

model = burgers.burgers(initial_data = burgers.initial_travelling_wave(tau))
source = burgers.stiff_source(tau, beta)
sim = simulation(model, interval, weno3_upwind, imex433(source), outflow)
sim.evolve(0.56*L)
q_exact = lambda x, t : 1/(1+numpy.exp(-(x-beta*t)/tau))
pyplot.figure()
pyplot.plot(x_exact, q_exact(x_exact, sim.t))
pyplot.plot(sim.coordinates, sim.q[0,:], 'kx')
pyplot.xlim(-L, L)
pyplot.ylim(-0.1, 1.1)
pyplot.xlabel(r"$x$")
pyplot.ylabel(r"$q$")
pyplot.title(r"Travelling wave, $\tau={}$".format(tau))
pyplot.show()


#Npoints_all = [50, 100, 200]
#tau3 = 0.01
#model3 = burgers.burgers(initial_data = burgers.initial_travelling_wave(tau3))
#source3 = burgers.stiff_source(tau3, beta)
#q_exact3 = lambda x, t : 1/(1+numpy.exp(-(x-beta*t)/tau3))
#pyplot.figure()
#t_end = 0.56*L
#pyplot.plot(x_exact, q_exact3(x_exact, t_end))
#for Npoints in Npoints_all:
#    interval3 = grid([-L, L], Npoints, Ngz)
#    sim3 = simulation(model3, interval, weno3_upwind, imex433(source3), outflow)
#    sim3.evolve(t_end)
#    pyplot.plot(sim3.coordinates, sim3.q[0,:], 'x--', mew=2, lw=2, label="{} points".format(Npoints))
#pyplot.xlim(-L, L)
#pyplot.ylim(-0.1, 1.1)
#pyplot.xlabel(r"$x$")
#pyplot.ylabel(r"$q$")
#pyplot.legend(loc="upper left")
#pyplot.title(r"Travelling wave, $\tau={}$".format(tau3))
#pyplot.show()
