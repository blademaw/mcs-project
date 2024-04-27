#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from model import *

# from joblib import Parallel, delayed


def generate_baseline():
    return BaselineModel(
            k=3,
            timestep=0.25,
            movement_dist=lambda : np.random.lognormal(-1999/4000000, np.log(1.001)),
            sigma_h_arr=np.array([19, 19, 19]),
            sigma_v_arr=np.array([.5, .5, .5]),
            K_v_arr=np.array([1000, 1000, 1000]),
            patch_densities=np.array([1/3, 1/3, 1/3]),
            phi_v_arr=np.array([.3, .3, .3]),
            beta_hv_arr=np.array([.33, .33, .33]),
            beta_vh_arr=np.array([.33, .33, .33]),
            nu_v_arr=np.array([.1, .1, .1]),
            mu_v_arr=np.array([1/14, 1/14, 1/14]),
            r_v_arr=np.array([.3-(1/14), .3-(1/14), .3-(1/14)]),
            num_locations=300,
            edge_prob=.03,
            num_agents=1500,
            initial_infect_proportion=.005,
            mu_h_dist=lambda : np.random.lognormal(-1.79226, np.log(1.001)),
            nu_h_dist=lambda : np.random.lognormal(-1.60994, np.log(1.001)),
            total_time=200,
            mosquito_timestep=.005
        )

def generate_baseline_low():
    return BaselineModel(
            k=3,
            timestep=0.25,
            movement_dist=lambda : np.random.lognormal(.01, .001),
            sigma_h_arr=np.array([19, 19, 19]),
            sigma_v_arr=np.array([.5, .5, .5]),
            K_v_arr=np.array([1000, 1000, 1000]),
            patch_densities=np.array([1/3, 1/3, 1/3]),
            phi_v_arr=np.array([.3, .3, .3]),
            beta_hv_arr=np.array([.33, .33, .33]),
            beta_vh_arr=np.array([.33, .33, .33]),
            nu_v_arr=np.array([.1, .1, .1]),
            mu_v_arr=np.array([1/14, 1/14, 1/14]),
            r_v_arr=np.array([.3-(1/14), .3-(1/14), .3-(1/14)]),
            num_locations=300,
            edge_prob=.03,
            num_agents=1500,
            initial_infect_proportion=.005,
            mu_h_dist=lambda : np.random.lognormal(1/6, .001),
            nu_h_dist=lambda : np.random.lognormal(1/5, .001),
            total_time=200,
            mosquito_timestep=.005
        )

def generate_low():
    return BaselineModel(
            k=3,
            movement_dist=lambda : np.random.lognormal(-4.60567, np.log(1.001)),
            timestep=0.25,
            sigma_h_arr=np.array([5, 19, 30]),
            sigma_v_arr=np.array([.5, .5, .5]),
            K_v_arr=np.array([750, 1500, 3750]),
            patch_densities=np.array([1/2, 1/3, 1/6]),
            phi_v_arr=np.array([.3, .3, .3]),
            beta_hv_arr=np.array([.33, .33, .33]),
            beta_vh_arr=np.array([.33, .33, .33]),
            nu_v_arr=np.array([.1, .1, .1]),
            mu_v_arr=np.array([1/14, 1/14, 1/14]),
            r_v_arr=np.array([.3-(1/14), .3-(1/14), .3-(1/14)]),
            num_locations=300,
            edge_prob=.03,
            num_agents=1500,
            initial_infect_proportion=.005,
            mu_h_dist=lambda : np.random.lognormal(-1.79226, np.log(1.001)),
            nu_h_dist=lambda : np.random.lognormal(-1.60994, np.log(1.001)),
            total_time=200,
            mosquito_timestep=.005
        )


# NOTE: parameters have been verified.
if __name__ == "__main__":
    reps = 50
    dist = []

    if reps == 1:
        model = generate_baseline()
        # model = generate_low()
        temporal, res = model.run(with_progress=True)
        # var = "patch1"
        #
        # plt.plot([sir[0] for sir in temporal[var]], label="S")
        # plt.plot([sir[1] for sir in temporal[var]], label="E")
        # plt.plot([sir[2] for sir in temporal[var]], label="I")
        # plt.show()

        print("Patch ticks:", temporal["patch_ticks"])

        print("Time of epidemic peak:")
        for k in range(3):
            print(f"Patch {k+1}:", np.argmax(temporal["num_infected"][k]))

        print("Estimated R0:")
        for k in range(3):
            print(f"Patch {k+1}:", np.argmax(temporal["r0"][k]))

        print(f"Number of total movements: {temporal['num_movements']}")
        print(f"\tAvg: {temporal['num_movements']/800}")

        print("Average number of days in:")
        # print(f"\tS: {(1/4)*temporal['total_time_in_state'][0]/1500}")
        print(f"\tE: {(1/4)*temporal['total_time_in_state'][1]/temporal['total_exposed']}")
        print(f"\tI: {(1/4)*temporal['total_time_in_state'][2]/temporal['total_infected']}")
        # print(f"\tR: {(1/4)*temporal['total_time_in_state'][3]/temporal['total_recovered']}")

        plt.plot(temporal["num_infected"][0], label="Patch 1")
        plt.plot(temporal["num_infected"][1], label="Patch 2")
        plt.plot(temporal["num_infected"][2], label="Patch 3")
        plt.legend()
        plt.show()

        print(res)
    else:
        patch_time_peaks = [[], [], []]

        for _ in tqdm(range(reps)):
            # model = generate_baseline()
            model = generate_low()
            temporal, res = model.run(with_progress=False)
            # print("Peak time p1:", np.argmax(temporal["num_infected"][0]))
            for i in range(3):
                patch_time_peaks[i].append(np.argmax(temporal["num_infected"][i])/4)
            dist.append(res[0])

        print("Number of infected hosts throughout simulation:")
        print(dist)
        dist = np.array(dist)
        print("Average number of infected hosts:", dist.mean(), "±", dist.std())

        print("\nTiming of peak epidemic per patch:")
        for k in range(3):
            print(f"\t{k+1}: {np.mean(patch_time_peaks[k])} ± {np.std(patch_time_peaks[k])}")
        temp = np.array(patch_time_peaks).flatten()
        print(f"\tTotal: {temp.mean()} ± {temp.std()}")

        plt.title("Distribution of total number of infections")
        # plt.hist(dist, bins=100, range=(0, 1500), density=True)
        sns.kdeplot(dist, bw_adjust=.75, color='black')
        plt.xlim(0, 1500)
        plt.xlabel("Total infected")
        plt.ylabel("Frequency")
        plt.show()


        plt.title("Timing of epidemic peak for patches")
        sns.kdeplot(patch_time_peaks[0], bw_adjust=.5, color="steelblue", label="Patch A")
        sns.kdeplot(patch_time_peaks[1], bw_adjust=.5, color="green", label="Patch B")
        sns.kdeplot(patch_time_peaks[2], bw_adjust=.5, color="red", linestyle="dashed", label="Patch C")
        plt.xlabel("Time of peak")
        plt.ylabel("Frequency")
        plt.xlim(0, 200)
        plt.show()

