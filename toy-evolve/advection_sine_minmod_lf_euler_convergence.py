# Advection test evolution: convergence test

from models import advection
from bcs import periodic
from simulation import simulation
from methods import minmod_lf
from rk import euler
from grid import grid
import numpy
from matplotlib import pyplot

Ngz = 2
Npoints_all = 40 * 2**numpy.arange(6)
dx_all = 1 / Npoints_all
errors = numpy.zeros((3,len(dx_all)))
for i, Npoints in enumerate(Npoints_all):
    print(Npoints)
    interval = grid([-0.5, 0.5], Npoints, Ngz)
    model = advection.advection(v=1, 
                                initial_data = advection.initial_sine(period=1))
    sim = simulation(model, interval, minmod_lf, euler, periodic)
    sim.evolve(1.0)
    errors[0, i] = sim.error_norm(1)
    errors[1, i] = sim.error_norm(2)
    errors[2, i] = sim.error_norm('inf')

norm_string = ("1", "2", "\infty")
fig = pyplot.figure(figsize=(12,6))
ax = fig.add_subplot(111)
for norm in range(3):
    p = numpy.polyfit(numpy.log(dx_all), numpy.log(errors[norm,:]), 1)
    ax.loglog(dx_all, errors[norm,:], 'x',
              label=r"$\| Error \|_{}$".format(norm_string[norm]))
    ax.loglog(dx_all, numpy.exp(p[1])*dx_all**p[0],
              label=r"$\propto \Delta x^{{{:.3f}}}$".format(p[0]))
ax.set_xlabel(r"$\Delta x$")
ax.set_ylabel("Error")
ax.legend(loc= "upper left")
pyplot.title("Advection, sine, L-F, Minmod, Euler")
pyplot.show()
    