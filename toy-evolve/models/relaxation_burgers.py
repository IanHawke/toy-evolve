import numpy

class relaxation_burgers(object):
    def __init__(self, initial_data, a=1.0):
        assert(a>0.0), "As characteristic speeds are \pm \sqrt(a), need a>0."
        self.a = a
        self.A_flux = numpy.array([[0.0,1.0],[self.a,0.0]])
        self.Nvars = 2
        self.flux = lambda q : numpy.dot(self.A_flux, q)
        self.initial_data = initial_data
        
    def riemann_problem_flux(self, q_L, q_R):
        """
        Principal part is a linear system: hand-calculate using Roe formula
        """
        f = numpy.zeros_like(q_L)
        for i in range(q_L.shape[1]):
            uL, vL = q_L[:,i]
            uR, vR = q_R[:,i]
            f[0,i] = 0.5*(       (vL+vR) - numpy.sqrt(self.a)*(uR-uL))
            f[1,i] = 0.5*(self.a*(uL+uR) - numpy.sqrt(self.a)*(vR-vL))
        return f
        
def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((2,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((2,len(x))))

def relaxation_source(tau):
    """
    See LeVeque section 17.17.
    """
    def source(q):
        s = numpy.zeros_like(q)
        s[1,:] = 1/tau * (q[0,:]**2/2 - q[1,:])
        return s
    return source
