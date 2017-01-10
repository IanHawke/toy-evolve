import numpy
from matplotlib import pyplot

class simulation(object):
    def __init__(self, model, grid, rhs, timestepper, bcs, cfl=0.5):
        self.model = model
        self.Nvars = model.Nvars
        self.grid = grid
        self.rhs = rhs
        self.timestepper = timestepper
        self.bcs = bcs
        self.cfl = cfl
        self.dx = grid.dx
        self.dt = cfl * self.dx
        self.coordinates = grid.coordinates()
        self.q0 = model.initial_data(self.coordinates)
        self.q0 = self.q0.reshape((self.Nvars, 
                                   self.grid.Npoints+2*self.grid.Ngz))
        self.q = self.q0.copy()
        self.t = 0
        
    def evolve_step(self, t_end):
        if self.t + self.dt > t_end:
            self.dt = t_end - self.t
        self.q = self.timestepper(self, self.q)
        self.q = self.bcs(self.q, self.grid.Npoints, self.grid.Ngz)
        self.t += self.dt

    def evolve(self, t_end):
        while self.t < t_end:
            self.evolve_step(t_end)

    def plot_scalar(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        x = self.grid.interior_coordinates()
        ax.plot(x, self.q[0, 
                              self.grid.Ngz:self.grid.Ngz+self.grid.Npoints])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q$")
        ax.set_xlim(self.grid.interval[0],self.grid.interval[1])
        qmax = numpy.max(self.q[0,:])
        qmin = numpy.min(self.q[0,:])
        dq = qmax - qmin
        ax.set_ylim(qmin-0.05*dq, qmax+0.05*dq)
        pyplot.title(r"Solution at $t={}$".format(self.t))
        return fig
        
    def plot_scalar_vs_initial(self):
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        x = self.grid.interior_coordinates()
        ax.plot(x, self.q[0, 
                          self.grid.Ngz:self.grid.Ngz+self.grid.Npoints],
                'b-', label="Evolved")
        ax.plot(x, self.q0[0, 
                          self.grid.Ngz:self.grid.Ngz+self.grid.Npoints],
                'g--', label="Initial")
        ax.legend(loc="upper left")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$q$")
        ax.set_xlim(self.grid.interval[0],self.grid.interval[1])
        qmax = numpy.max(self.q[0,:])
        qmin = numpy.min(self.q[0,:])
        dq = qmax - qmin
        ax.set_ylim(qmin-0.05*dq, qmax+0.05*dq)
        pyplot.title(r"Solution at $t={}$".format(self.t))
        return fig
        
    def error_norm(self, norm):
        if norm=='inf' or norm==numpy.inf:
            return numpy.max(numpy.abs(self.q[0,:] - self.q0[0,:]))
        else:
            return (numpy.sum(numpy.abs(self.q[0,:] - self.q0[0,:])**norm)*self.dx)**(1/norm)
        