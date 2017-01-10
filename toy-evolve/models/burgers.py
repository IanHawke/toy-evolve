import numpy

class burgers(object):
    def __init__(self, initial_data):
        self.Nvars = 1
        self.flux = lambda q : 0.5 * q**2
        self.initial_data = initial_data
        
    def riemann_problem_flux(self, q_L, q_R):
        f = numpy.zeros_like(q_L)
        for i, (qL, qR) in enumerate(zip(q_L[0,:], q_R[0,:])):
            if qL < qR:
                if qL > 0:
                    f[0,i] = 0.5*qL**2
                elif qL * qR < 0:
                    f[0,i] = 0
                else:
                    f[0,i] = 0.5*qR**2
            else:
                if qL + qR > 0:
                    f[0,i] = 0.5*qL**2
                else:
                    f[0,i] = 0.5*qR**2
        return f
        
def initial_sine(period=1):
    return lambda x : numpy.sin(2*numpy.pi*period*x)

def initial_square():
    return lambda x : numpy.where(numpy.abs(x) < 0.25,
                                  numpy.ones_like(x),
                                  numpy.zeros_like(x))

def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql*numpy.ones_like(x),
                                  qr*numpy.ones_like(x))

def initial_travelling_wave(tau):
    return lambda x : 1/(1+numpy.exp(-x/tau))

def stiff_source(tau, beta):
    """
    See LeVeque p401 eq 17.74
    
    Note: max of d source / dq is (beta**2 - beta + 1) / (3 tau).
    """
    def source(q):
        return 1/tau * q * (1 - q) * (q - beta)
    return source