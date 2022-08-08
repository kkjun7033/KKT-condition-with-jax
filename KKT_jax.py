import numpy as np
import pandas as pd
import random
import cvxpy as cp

from jax import vjp
import jax.numpy as jnp
from jaxopt import implicit_diff



def f(x, theta):
    return jnp.sum(jnp.dot(obj, x))

def H(x):   
    return jnp.dot(lhs_inequal, x) - rhs_inequal 

def G(x):   
    return jnp.dot(lhs_equal, x) - rhs_equal

grad = jax.grad(f)

def F(x, thet):
    z, dual1, dual2 = x 
    
    h_y, h_vjp = vjp(H, z) 
    g_y, g_vjp = vjp(G, z)
    
    stationarity = (grad(z, thet) + h_vjp(dual1)[0] + g_vjp(dual2)[0])
    primal_feasability = H(z)
    comp_slackness = G(z)*dual2

    return stationarity, primal_feasability, comp_slackness

@implicit_diff.custom_root(F)
def solver(init_x, thet):
    del init_x 
    xcp = cp.Variable(dims)
    prob = cp.Problem(cp.Minimize(obj@xcp), [lhs_inequal@xcp<=rhs_inequal.reshape(link_dim), lhs_equal@xcp==rhs_equal.reshape(node_dim), xcp>=0])
    prob.solve()
    result_x = xcp.value
    result_dual1 = prob.constraints[0].dual_value
    result_dual2 = prob.constraints[1].dual_value    
    return jnp.array(result_x), jnp.array(result_dual1), jnp.array(result_dual2) 


dx_dtheta = jax.jacobian(solver, argnums=1)(init_x, theta)