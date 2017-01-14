import numpy
from scipy.optimize import fsolve

def euler(simulation, q):
    dt = simulation.dt
    rhs = simulation.rhs
    return q + dt * rhs(q, simulation)

def rk2(simulation, q):
    dt = simulation.dt
    rhs = simulation.rhs
    q1 = q + dt * rhs(q, simulation)
    q1 = simulation.bcs(q1, simulation.grid.Npoints, simulation.grid.Ngz)
    return 0.5 * (q + q1 + dt * rhs(q1, simulation))

def rk3(simulation, q):
    dt = simulation.dt
    rhs = simulation.rhs
    q1 = q + dt * rhs(q, simulation)
    q1 = simulation.bcs(q1, simulation.grid.Npoints, simulation.grid.Ngz)
    q2 = (3 * q + q1 + dt * rhs(q1, simulation)) / 4
    q2 = simulation.bcs(q2, simulation.grid.Npoints, simulation.grid.Ngz)
    return (q + 2 * q2 + 2 * dt * rhs(q2, simulation)) / 3

def rk_euler_split(rk_method, source):
    def timestepper(simulation, q):
        qstar = rk_method(simulation, q)
        return qstar + simulation.dt * source(qstar)
    return timestepper

def rk_backward_euler_split(rk_method, source):
    def timestepper(simulation, q):
        qstar = rk_method(simulation, q)
        def residual(qguess):
            return qguess - qstar.ravel() - simulation.dt*source(qguess).ravel()
        q_initial_guess = qstar + 0.5*simulation.dt*source(qstar)
        qnext = fsolve(residual, q_initial_guess.ravel())
        return numpy.reshape(qnext, q.shape)
    return timestepper

def imex222(source):
    def timestepper(simulation, q):
        gamma = 1 - 1/numpy.sqrt(2)
        dt = simulation.dt
        rhs = simulation.rhs
        def residual1(qguess):
            return qguess - q.ravel() - dt * gamma * source(qguess.reshape(q.shape)).ravel()
        qguess = q.copy()
        q1 = fsolve(residual1, qguess.ravel()).reshape(q.shape)
        q1 = simulation.bcs(q1, simulation.grid.Npoints, simulation.grid.Ngz)
        k1 = rhs(q1, simulation)
        source1 = source(q1)
        def residual2(qguess):
            return qguess - q.ravel() - dt * (k1.ravel() + (1 - 2*gamma)*source1.ravel() + gamma*source(qguess.reshape(q.shape)).ravel())
        q2 = fsolve(residual2, q1.copy().ravel()).reshape(q.shape)
        q2 = simulation.bcs(q2, simulation.grid.Npoints, simulation.grid.Ngz)
        k2 = rhs(q2, simulation)
        source2 = source(q2)
        return q + simulation.dt * (k1 + k2 + source1 + source2) / 2
    return timestepper

def imex433(source):
    def timestepper(simulation, q):
        alpha = 0.24169426078821
        beta = 0.06042356519705
        eta = 0.12915286960590
        dt = simulation.dt
        rhs = simulation.rhs
        def residual1(qguess):
            return qguess - q.ravel() - dt * alpha * source(qguess)
        qguess = q.copy() + 0.5*dt*source(q)
        q1 = fsolve(residual1, qguess.ravel()).reshape(q.shape)
        q1 = simulation.bcs(q1, simulation.grid.Npoints, simulation.grid.Ngz)
#        k1 = rhs(q1, simulation)
        source1 = source(q1)
        def residual2(qguess):
            return qguess - q.ravel() - dt * (-alpha*source1.ravel() + alpha*source(qguess))
        q2 = fsolve(residual2, q1.copy().ravel()).reshape(q.shape)
        q2 = simulation.bcs(q2, simulation.grid.Npoints, simulation.grid.Ngz)
        k2 = rhs(q2, simulation)
        source2 = source(q2)
        def residual3(qguess):
            return qguess - q.ravel() - dt * (k2.ravel() + (1 - alpha)*source2.ravel() + alpha*source(qguess))
        q3 = fsolve(residual3, q2.copy().ravel()).reshape(q.shape)
        q3 = simulation.bcs(q3, simulation.grid.Npoints, simulation.grid.Ngz)
        k3 = rhs(q3, simulation)
        source3 = source(q3)
        def residual4(qguess):
            return qguess - q.ravel() - dt * ((k2.ravel() + k3.ravel())/4 + beta*source1.ravel() + eta*source2.ravel() + (1/2-beta-eta-alpha)*source3.ravel() + alpha*source(qguess))
        q4 = fsolve(residual4, q3.copy().ravel()).reshape(q.shape)
        q4 = simulation.bcs(q4, simulation.grid.Npoints, simulation.grid.Ngz)
        k4 = rhs(q4, simulation)
        source4 = source(q4)
        return q + simulation.dt * (k2+k3+4*k4 + source2+source3+4*source4) / 6
    return timestepper
