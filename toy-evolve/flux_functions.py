import numpy

def lax_friedrichs(q_minus, q_plus, simulation):
    alpha = simulation.dx / simulation.dt
    flux = numpy.zeros_like(q_minus)
    f_minus = simulation.model.flux(q_minus)
    f_plus = simulation.model.flux(q_plus)
    
    flux[:, 1:-1] = 0.5 * ( (f_plus[:,0:-2] + f_minus[:,1:-1]) + \
                    alpha * (q_plus[:,0:-2] - q_minus[:,1:-1]) )
    
    return flux

def upwind(q_minus, q_plus, simulation):
    flux = numpy.zeros_like(q_minus)
    flux[:, 1:-1] = simulation.model.riemann_problem_flux(q_plus[:,0:-2], 
                                                          q_minus[:, 1:-1])
    return flux
    