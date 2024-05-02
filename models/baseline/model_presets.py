#!/usr/bin/env python3

import numpy as np

BASELINE_MODEL={
    "k" : 3,
    "timestep" : 0.25,
    "movement_dist" : lambda : np.random.lognormal(-1999/4000000, np.sqrt(np.log(1.001))),
    "sigma_h_arr" : np.array([19, 19, 19]),
    "sigma_v_arr" : np.array([.5, .5, .5]),
    "K_v_arr" : np.array([1000, 1000, 1000]),
    "patch_densities" : np.array([1/3, 1/3, 1/3]),
    "phi_v_arr" : np.array([.3, .3, .3]),
    "beta_hv_arr" : np.array([.33, .33, .33]),
    "beta_vh_arr" : np.array([.33, .33, .33]),
    "nu_v_arr" : np.array([.1, .1, .1]),
    "mu_v_arr" : np.array([1/14, 1/14, 1/14]),
    "r_v_arr" : np.array([.3-(1/14), .3-(1/14), .3-(1/14)]),
    "num_locations" : 300,
    "edge_prob" : .03,
    "num_agents" : 1500,
    "initial_infect_proportion" : .005,
    "mu_h_dist" : lambda : np.random.lognormal(-1.80944, .188062),
    "nu_h_dist" : lambda : np.random.lognormal(-1.62178, .157139),
    "total_time" : 200,
    "mosquito_timestep" : .005
}


HIGH_MOVEMENT = {
    "k" : 3,
    "movement_dist" : lambda : np.random.lognormal(-1999/4000000, np.sqrt(np.log(1.001))),
    "timestep" : 0.25,
    "sigma_h_arr" : np.array([5, 19, 30]),
    "sigma_v_arr" : np.array([.5, .5, .5]),
    "K_v_arr" : np.array([750, 1500, 3750]),
    "patch_densities" : np.array([1/2, 1/3, 1/6]),
    "phi_v_arr" : np.array([.3, .3, .3]),
    "beta_hv_arr" : np.array([.33, .33, .33]),
    "beta_vh_arr" : np.array([.33, .33, .33]),
    "nu_v_arr" : np.array([.1, .1, .1]),
    "mu_v_arr" : np.array([1/14, 1/14, 1/14]),
    "r_v_arr" : np.array([.3-(1/14), .3-(1/14), .3-(1/14)]),
    "num_locations" : 300,
    "edge_prob" : .03,
    "num_agents" : 1500,
    "initial_infect_proportion" : .005,
    "mu_h_dist" : lambda : np.random.lognormal(-1.80944, .188062),
    "nu_h_dist" : lambda : np.random.lognormal(-1.62178, .157139),
    "total_time" : 200,
    "mosquito_timestep" : .005
}
