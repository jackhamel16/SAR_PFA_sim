import numpy as np
import pytest

import SAR

def test_projection_func_flat():
    tol = 1e-8
    x_min, x_max = -5, 5
    test_p = SAR.compute_projection_func(0, x_min, x_max, lambda x, y: 1 if np.abs(y) < 1 else 0, 10)
    
    sol = 10.0
    assert np.abs(test_p - sol) < 1e-8
    
def test_projection_func_linear():
    tol = 1e-8
    x_min, x_max = 0, 5
    test_p = SAR.compute_projection_func(0, x_min, x_max, lambda x, y: x, 10)
    
    sol = 25/2
    assert np.abs(test_p - sol) < 1e-8
    
def test_projection_func_circle():
    num_integral_terms = 10000
    patch_radius = 500
    reflective_radius = patch_radius / 4
    
    def projection_func(y):
        reflectivity_func = lambda x, y : 2 * SAR.reflectivity_circ_rect(x, y, reflective_radius)
        x_bound = np.sqrt(patch_radius**2 - y**2)
        x_min, x_max = -1*x_bound, x_bound
        num_terms = int(num_integral_terms * x_bound / patch_radius) #this will ensure I have uniformly space samples as y changes
        return(SAR.compute_projection_func(y, x_min, x_max, reflectivity_func, num_terms))
    
    y1, y2 = 0.0, 100.0
    test1 = projection_func(y1)
    test2 = projection_func(y2)
    
    sol1, sol2 = 500.0, 300.0
    
    print(np.abs(test1 - sol1)/np.abs(sol1))
    assert np.abs(test1 - sol1)/np.abs(sol1) < 1e-8
    assert np.abs(test2 - sol2)/np.abs(sol2) < 1e-3

def test_compute_processed_return():
    chirp_A = 2
    f0 = 2.2 / 2 / np.pi # Hz
    T = 1 # seconds
    desired_res = 1 # meters
    R = 5 # meters
    patch_edge_length = 3 # m
    patch_center_coords = [0, 0, 0]
    params = SAR.sim_params(chirp_A, f0, T, R, desired_res, desired_res, patch_edge_length, patch_center_coords)
    params.c = 5.1
    params.alpha = 3
    t = 15
    y_min, y_max = -1, 2.5
    num_integral_terms = 1000
    proj_func = lambda y: y
    
    test_return = SAR.compute_processed_return(t, y_min, y_max, proj_func, params, num_integral_terms)
    
    sol = -0.030781077122546 - 0.043641058100540j # sol obtained by using trapz in matlab
    
    assert np.abs(test_return - sol < 1e-3)
    
def test_numerically_integrate():
    test = SAR.numerically_integrate(0,3,1000,lambda x: x**2)
    
    sol = 9.0
    assert np.abs(test - sol) < 1e-3
    
return_code = pytest.main(['-v'])