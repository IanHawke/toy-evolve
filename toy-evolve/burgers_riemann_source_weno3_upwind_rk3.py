# Burgers test evolution: just one

import numpy
from models import burgers
from bcs import outflow
from simulation import simulation
from methods import weno3, slope_minmod, rea_method_source, minmod_lf, weno3_upwind
from flux_functions import upwind, lax_friedrichs
from rk import rk2, rk3, rk_euler_split, rk_backward_euler_split, imex222
from grid import grid
from matplotlib import pyplot

Ngz = 3
Npoints = 200
tau = 0.05
beta = 0.8
L = 1
interval = grid([-L, L], Npoints, Ngz)
model = burgers.burgers(initial_data = burgers.initial_riemann(0, 1))
model = burgers.burgers(initial_data = burgers.initial_travelling_wave(tau))
source = burgers.stiff_source(tau, beta)
weno_lf_source = rea_method_source(weno3, upwind, source)
rk2_euler_split = rk_euler_split(rk2, source)
rk3_be_split = rk_backward_euler_split(rk3, source)
#minmod_lf_source = rea_method_source(slope_minmod, lax_friedrichs, source)
#sim = simulation(model, interval, weno3_upwind, rk3_be_split, outflow)
sim = simulation(model, interval, weno3_upwind, imex222(source), outflow)
sim.evolve(0.56*L)

sim.plot_scalar_vs_initial()
pyplot.show()

q_exact = lambda x, t : 1/(1+numpy.exp(-(x-beta*t)/tau))
x_exact = numpy.linspace(-L,L,1000)
pyplot.figure()
pyplot.plot(x_exact, q_exact(x_exact, sim.t))
pyplot.plot(sim.coordinates, sim.q[0,:], 'kx')
pyplot.xlim(-L, L)
pyplot.ylim(-0.1, 1.1)
pyplot.xlabel(r"$x$")
pyplot.ylabel(r"$q$")
pyplot.title(r"Travelling wave, $\tau={}$".format(tau))
pyplot.show()


tau2 = 0.01
model2 = burgers.burgers(initial_data = burgers.initial_travelling_wave(tau2))
source2 = burgers.stiff_source(tau2, beta)
rk3_be_split2 = rk_backward_euler_split(rk3, source2)
#sim2 = simulation(model2, interval, weno3_upwind, rk3_be_split2, outflow)
sim2 = simulation(model2, interval, weno3_upwind, imex222(source2), outflow)
sim2.evolve(0.56*L)
q_exact2 = lambda x, t : 1/(1+numpy.exp(-(x-beta*t)/tau2))
pyplot.figure()
pyplot.plot(x_exact, q_exact2(x_exact, sim2.t))
pyplot.plot(sim2.coordinates, sim2.q[0,:], 'kx')
pyplot.xlim(-L, L)
pyplot.ylim(-0.1, 1.1)
pyplot.xlabel(r"$x$")
pyplot.ylabel(r"$q$")
pyplot.title(r"Travelling wave, $\tau={}$".format(tau2))
pyplot.show()


Npoints_all = [50, 100, 200]
tau3 = 0.01
model3 = burgers.burgers(initial_data = burgers.initial_travelling_wave(tau3))
source3 = burgers.stiff_source(tau3, beta)
rk3_be_split3 = rk_backward_euler_split(rk3, source3)
q_exact3 = lambda x, t : 1/(1+numpy.exp(-(x-beta*t)/tau3))
pyplot.figure()
t_end = 0.56*L
pyplot.plot(x_exact, q_exact3(x_exact, t_end))
for Npoints in Npoints_all:
    interval3 = grid([-L, L], Npoints, Ngz)
#    sim3 = simulation(model3, interval, weno3_upwind, rk3_be_split3, outflow)
    sim3 = simulation(model3, interval, weno3_upwind, imex222(source3), outflow)
    sim3.evolve(t_end)
    pyplot.plot(sim3.coordinates, sim3.q[0,:], 'x--', mew=2, lw=2, label="{} points".format(Npoints))
pyplot.xlim(-L, L)
pyplot.ylim(-0.1, 1.1)
pyplot.xlabel(r"$x$")
pyplot.ylabel(r"$q$")
pyplot.legend(loc="upper left")
pyplot.title(r"Travelling wave, $\tau={}$".format(tau3))
pyplot.show()
