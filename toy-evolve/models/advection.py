import numpy

class advection(object):
    def __init__(self, initial_data, v=1):
        self.Nvars = 1
        self.v = v
        self.flux = lambda q : v * q
        self.initial_data = initial_data
        
    def riemann_problem_flux(self, q_L, q_R):
        return self.flux(q_L)

def initial_sine(period=1):
    return lambda x : numpy.sin(2*numpy.pi*period*x)

def initial_square():
    return lambda x : numpy.where(numpy.abs(x) < 0.25,
                                  numpy.ones_like(x),
                                  numpy.zeros_like(x))
