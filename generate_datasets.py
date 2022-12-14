import pdb
import os
import functools
import numpy as np
import matplotlib.pyplot as plt
import sdeint

from datetime import datetime
from time import time
from abc import abstractmethod
from scipy.integrate import odeint

from utils import *
from generation_parameters import *
from plotting import compare_3d_datasets_one_plot, plot_time_series


# number of time steps to skip in order to avoid transient behavior
# so that generated data only contains the attracting sets
SKIP_TRANS = 8000
# time step size for numerical integration
STEP_SIZE = 0.002  # 0.05 for BN, 0.01 for L63
# enable stochastic differential equation solver
SDE = False

STANDARDIZE = True


def plot_param_dynamics_from_dict(params_at_all_t: dict, savepath: str):

    plt.figure(figsize=(20, 12))
    plt.title("Parameter evolution")
    param_values = {}
    for i, idx in enumerate(params_at_all_t):
        param_dict = params_at_all_t[idx]
        for j, name in enumerate(param_dict):
            if not name in param_values.keys():
                param_values[name] = []
            else:
                param_values[name].append(param_dict[name])

    for i, name in enumerate(param_values):
        scaled_values = (
            1.0
            if param_values[name][0] == 0.0
            else param_values[name] / param_values[name][0]
        )
        plt.plot(scaled_values, label=name)
    plt.xlabel("time steps")
    plt.ylabel("relative change")
    plt.legend()
    plt.savefig(os.path.join(savepath, "parameter_evolution.pdf"))


class DynamicalSystem:
    """
    General model of an (time-dependent) dynamical system.
    """

    def __init__(
        self,
        params_init: dict,
        schedules: dict,
        eta_param=0,
        eta_process=0,
        time_step=0.01,
    ):

        self.time_step = time_step
        self.params_init = params_init
        self.schedules = schedules
        # parameter noise
        self.eta_param = eta_param
        # process noise
        self.eta_process = eta_process
        # fixed parameters for stationary data generation
        self.params_stat = None

        if not self.params_init.keys() == self.schedules.keys():
            raise Exception("Error! Parameters and schedules do not match!")

        self.all_params = None

    def get_params(self, t):
        """
        Computes parameters of system at time t.
        """
        # time to be skipped to avoid transients
        t_skip = SKIP_TRANS * self.time_step

        if t >= t_skip:
            if not self.all_params is None:
                # if parameters have been precomputed
                idx = int((t - t_skip) / self.time_step)
                return self.all_params[idx]
            else:
                # otherwise, compute during generation of the trajectory (very slow)
                return get_params_from_schedule(
                    self.params_init, self.schedules, t - t_skip, self.time_step
                )
        else:
            return self.params_init

    def set_all_params(self, params):
        """
        Precompute parameters for all time steps. Returns dictionary
        containing all parameters at each time step.
        """
        self.all_params = params

    @abstractmethod
    def equations_determ(self):
        raise NotImplementedError

    @abstractmethod
    def equations_noise(self):
        raise NotImplementedError

    def model_deterministic(self, state, t):
        """
        Deterministic part of the system.
        """
        if self.params_stat is None:
            params = self.get_params(t)
        else:
            params = self.params_stat
        state_new = self.equations_determ(state, params)
        assert type(state_new) == list
        return np.array(state_new)

    def model_noise(self, state, t):
        """
        Noise/Stochastic part of the system.
        """
        noise = self.equations_noise(state)
        assert type(noise) == list
        return np.array([noise]).T

    def generate_data(self, x0, n, t_stat=None):
        """
        Generates a free trajectory from the dynamical system.
        """
        # for stationary data generation with parameters
        # fixed at time t_stat
        if not t_stat is None:
            self.params_stat = self.get_params(t_stat + SKIP_TRANS * self.time_step)

        T = (n + SKIP_TRANS) * self.time_step
        t = np.arange(0.0, T, self.time_step)

        if SDE:
            X = sdeint.itoint(
                self.model_deterministic, self.model_noise, x0.reshape(-1), t
            )
        else:
            X = odeint(self.model_deterministic, x0.reshape(-1), t)
        data_gen = X[SKIP_TRANS:, :]

        self.params_stat = None
        return data_gen


class Lorenz63(DynamicalSystem):
    """
    Lorenz system with time-varying parameters and noise.
    """

    def __init__(
        self, params_init, schedules, eta_param=0, eta_process=0, time_step=0.01
    ) -> None:
        super().__init__(params_init, schedules, eta_param, eta_process, time_step)

    def equations_determ(self, state, params):
        """
        Deterministic part of the non-stationary Lorenz system.
        """
        x, y, z = state
        sigma = params["sigma"]
        rho = params["rho"]
        beta = params["beta"]
        # Lorenz equations (without noise term eta on parameters)
        x_new = sigma * (y - x)  # + eta * (y - x)
        y_new = x * (rho - z) - y  # + eta * x
        z_new = x * y - beta * z  # - eta * z
        return [x_new, y_new, z_new]

    def equations_noise(self, state):
        """
        Noise part of the non-stationary Lorenz system.
        """
        x, y, z = state
        # apply noise on both parameters and data
        x_noise = self.eta_param * (y - x) + self.eta_process
        y_noise = self.eta_param * x + self.eta_process
        z_noise = self.eta_param * (-z) + self.eta_process
        return [x_noise, y_noise, z_noise]


class BurstingNeuron(DynamicalSystem):
    def __init__(
        self, params_init, schedules, eta_param=0, eta_process=0, time_step=0.01
    ) -> None:
        super().__init__(params_init, schedules, eta_param, eta_process, time_step)

    def gate(self, V, V_h_, k_):
        return 1 / (1 + np.exp((V_h_ - V) / k_))

    def sigma(self, V):
        return 1 / (1 + 0.33 * np.exp(-0.0625 * V))

    def equations_determ(self, state, params):
        """
        Deterministic part of the non-stationary bursting neuron system.
        """
        V, n, h = state
        I = params["I"]
        C_m = params["C_m"]
        g_L = params["g_L"]
        E_L = params["E_L"]
        g_Na = params["g_Na"]
        E_Na = params["E_Na"]
        V_h_Na = params["V_h_Na"]
        k_Na = params["k_Na"]
        g_K = params["g_K"]
        E_K = params["E_K"]
        V_h_K = params["V_h_K"]
        k_K = params["k_K"]
        tau_n = params["tau_n"]  # tcK
        g_M = params["g_M"]
        V_h_M = params["V_h_M"]
        k_M = params["k_M"]
        tau_h = params["tau_h"]  # tcM
        g_NMDA = params["g_NMDA"]
        E_NMDA = params["E_NMDA"]
        # V_s_inf = params['V_s_inf']

        m_inf = 1 / (1 + np.exp((V_h_Na - V) / k_Na))
        n_inf = 1 / (1 + np.exp((V_h_K - V) / k_K))
        h_inf = 1 / (1 + np.exp((V_h_M - V) / k_M))

        # if V_s_inf==0:
        #     s_inf = 1 / (1 + 0.33 * np.exp(-0.0625 * V))
        # else:
        #     s_inf = 1 / (1 + 0.33 * np.exp(-0.0625 * V_s_inf))

        # flow field
        dv = (
            I
            - g_L * (V - E_L)
            - g_Na * m_inf * (V - E_Na)
            - g_K * n * (V - E_K)
            - g_M * h * (V - E_K)
            - g_NMDA * 1 / (1 + 0.33 * np.exp(-0.0625 * V)) * (V - E_NMDA)
        ) / C_m
        dn = (n_inf - n) / tau_n
        dh = (h_inf - h) / tau_h

        return [dv, dn, dh]

    def equations_noise(self, state):
        """
        Noise part of the non-stationary bursting neuron system.
        """
        x, y, z = state
        # not implemented ...
        return [0, 0, 0]


class Roessler(DynamicalSystem):
    def __init__(
        self, params_init, schedules, eta_param=0, eta_process=0, time_step=0.01
    ) -> None:
        super().__init__(params_init, schedules, eta_param, eta_process, time_step)

    def equations_determ(self, state, params):
        """
        Deterministic part of the non-stationary Rössler system.
        """
        x, y, z = state
        a = params["a"]
        b = params["b"]
        c = params["c"]
        x_new = -y - z
        y_new = x + a * y
        z_new = b + z * (x - c)
        return [x_new, y_new, z_new]

    def equations_noise(self, state):
        """
        Noise part of the non-stationary Rössler system.
        """
        x, y, z = state
        # not implemented ...
        return [0, 0, 0]


def plot_3D_trajectory(data, results_dir_, file_name, plot_bounds=None):
    linewidth = 0.5
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection="3d")
    ax.plot3D(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        "blue",
        linewidth=linewidth,
        label="generated trajectory",
    )
    ax.scatter3D(*data[0, :3], marker="o", color="green", s=50, label="initial state")
    ax.scatter3D(*data[-1, :3], marker="x", color="red", s=90, label="final state")

    if not plot_bounds is None:
        ax.set_xlim3d(*plot_bounds[0])
        ax.set_ylim3d(*plot_bounds[1])
        ax.set_zlim3d(*plot_bounds[2])
    plt.legend()
    plt.savefig(os.path.join(results_dir_, f"{file_name}.pdf"))
    plt.close()


# all available dynamical systems for data generation
SYSTEMS = {
    "Lorenz63": Lorenz63,
    "Bursting_Neuron": BurstingNeuron,
    "Rössler": Roessler,
}


def main():

    # for generating snapshot attractors
    generate_init_final = True
    # initial condition
    # x0 = np.array([.5, .5, .5])
    # x0 = np.array([-2.44694e+01,  3.86000e-02,  2.31000e-02])
    # number of time steps to generate
    n = 12000
    T = n * STEP_SIZE
    print(f"Final time T={T}, {n} steps, step size {STEP_SIZE}")

    # for all parameters, define their initial values and non-stationarities

    # some templates ...
    templ = "L63#2"

    ####################
    # non-stat L63 #1
    ####################
    if templ == "L63#1":
        system_name = "Lorenz63"
        x0 = np.random.randn(3)
        x0_test = np.random.randn(3)
        params_init = {
            "rho": 35.0,
            "sigma": 10.0,
            "beta": 8 / 3.0,
        }
        params_final = {
            "rho": 22.0,
            "sigma": 7.0,
            "beta": 1.87,
        }
        schedules = get_linear_schedules_from_final_params(
            params_init, params_final, T=T
        )

    ####################
    # non-stat L63 #2
    ####################
    elif templ == "L63#2":
        system_name = "Lorenz63"
        x0 = np.random.randn(3)
        x0_test = np.random.randn(3)
        params_init = {
            "rho": 160.0,
            "sigma": 10.0,
            "beta": 8 / 3.0,
        }
        params_final = params_init.copy()
        params_final["rho"] = 180.0
        schedules = get_linear_schedules_from_final_params(
            params_init, params_final, T=T
        )

    ####################
    # non-stat BN #1
    ####################
    elif templ == "BN#1":
        system_name = "Bursting_Neuron"
        # x0 = np.array([-2.44694e+01,  3.86000e-02,  2.31000e-02])
        x0 = np.random.rand(3) * np.array([-1.5, 0.01, 0.015])
        x0_test = np.random.rand(3) * np.array([-1.5, 0.01, 0.015])
        params_init = get_init_params(system_name)
        params_final = params_init.copy()
        schedules = schedules_all_stationary(
            params_init
        )  # all schedules are identity map
        params_init["g_NMDA"] = 9.25
        params_final["g_NMDA"] = 10.25
        schedules["g_NMDA"] = get_linear_schedule_from_final_param(
            params_init["g_NMDA"], params_final["g_NMDA"], T
        )

    else:
        ...
        exit(0)
        # own code ...

    # initialize dynamical system
    system = SYSTEMS[system_name](params_init, schedules, time_step=STEP_SIZE)
    # precompute parameters for all time steps up to time T
    params_all_t = get_params_all_time_steps(
        params_init, params_final, schedules, T, STEP_SIZE
    )
    system.set_all_params(params_all_t)

    print("Generating trajectory of", system_name)
    tic = time()
    data = system.generate_data(x0, n)
    data_test = system.generate_data(x0_test, n)
    if generate_init_final:
        print("Generating trajectory at t=0")
        data_init = system.generate_data(x0, n, t_stat=0)
        print(f"Generating trajectory at t={T}")
        data_final = system.generate_data(x0, n, t_stat=T)

    if STANDARDIZE and generate_init_final:
        (data, data_test, data_init, data_final) = standardize_several_datasets(
            [data, data_test, data_init, data_final]
        )
    elif STANDARDIZE:
        data = standardize(data)
    toc = time()

    print(f"Generated in {toc-tic:.2f} seconds")

    # save results
    results_dir = "generated_datasets"
    try:
        os.mkdir(results_dir)
    except:
        pass
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name_ = time_str + "_" + system_name
    results_dir_ = os.path.join(results_dir, name_)
    try:
        os.mkdir(results_dir_)
    except:
        pass

    # print log
    file = open(os.path.join(results_dir_, "log.txt"), "w")
    file.write(f"Generation log for {system_name}:\n\n")

    if not system.all_params is None:
        params_at_T = system.all_params[sorted(system.all_params.keys())[n]]
        plot_param_dynamics_from_dict(system.all_params, results_dir_)
    else:
        params_at_T = get_params_from_schedule(params_init, schedules, T, STEP_SIZE)
    for i, k in enumerate(params_at_T):
        p0 = params_init[k]
        pT = params_at_T[k]
        sched = schedules[k]
        file.write(f"Param name: {k}, init: {p0}, final: {pT}, schedule: {sched}\n")

    file.write("\n")
    file.write(f"Skipped {SKIP_TRANS} steps to avoid transients\n")
    file.write(f"Final time T={T}, {n} steps, step size {STEP_SIZE}\n")
    file.write(f"Initial condition: {x0}\n")
    file.write(f"Initial condition (test set): {x0_test}\n")
    file.write(f"Use of SDE integrator: {SDE}\n")
    file.close()

    # plot trajectories
    if n > 15000:
        # for better performance and visuals
        # if too many time steps where generated
        interv = int(np.round(n / 15000))
        # if too many time steps, plot arbitrary interval
        start = 64500
        stop = start + 1500
    else:
        interv = 1
        start = None
        stop = None

    plot_bounds = [
        [data[:, 0].min() * 1.1, data[:, 0].max() * 1.1],  # x bounds for plot
        [data[:, 1].min() * 1.1, data[:, 1].max() * 1.1],  # y bounds for plot
        [data[:, 2].min() * 1.1, data[:, 2].max() * 1.1],  # z bounds for plot
    ]

    plot_3D_trajectory(data, results_dir_, "gen_data_plot", plot_bounds)
    plot_time_series(
        data[start:stop],
        f"{system_name}, {n} time steps",
        savepath=results_dir_,
        name="gen_data_plot_each_dim",
    )
    plot_3D_trajectory(data_test, results_dir_, "gen_data_plot_test", plot_bounds)
    plot_time_series(
        data_test[start:stop],
        f"{system_name}, {n} time steps",
        savepath=results_dir_,
        name="gen_data_plot_each_dim_test",
    )
    if generate_init_final:
        plot_time_series(
            data_init[start:stop],
            f"{system_name} at t=0, {n} time steps",
            savepath=results_dir_,
            name="gen_data_each_dim_init_plot",
        )
        plot_time_series(
            data_final[start:stop],
            f"{system_name} at t={T}, {n} time steps",
            savepath=results_dir_,
            name="gen_data_each_dim_final_plot",
        )
    if generate_init_final:
        plot_3D_trajectory(data_init, results_dir_, "gen_data_plot_init", plot_bounds)
        plot_3D_trajectory(data_final, results_dir_, "gen_data_plot_final", plot_bounds)
        compare_3d_datasets_one_plot(
            data_init,
            data_final,
            label1="Gen. data t=0",
            label2="Gen. data t=T",
            alpha=1.0,
            name="comparison_one_plot_init_final",
            savepath=results_dir_,
            file_type=".pdf",
            linewidth=2.0,
        )

    # plot time-evolution of trajectory
    print("Plotting time evolution...")
    xval = data[::interv, 0]
    yval = data[::interv, 1]
    zval = data[::interv, 2]
    tval = np.arange(len(xval))
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    cax = ax.scatter(xval, yval, zval, cmap=plt.cm.viridis, c=tval, s=15)
    fig.colorbar(cax, shrink=0.6)
    plt.title("Time evolution: generated dataset", size=18)
    print("Saving at", results_dir_)
    plt.savefig(os.path.join(results_dir_, "time_evolution.pdf"))
    plt.close()
    xval = data_test[::interv, 0]
    yval = data_test[::interv, 1]
    zval = data_test[::interv, 2]
    tval = np.arange(len(xval))
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    cax = ax.scatter(xval, yval, zval, cmap=plt.cm.viridis, c=tval, s=15)
    fig.colorbar(cax, shrink=0.6)
    plt.title("Time evolution: generated dataset (test set)", size=18)
    plt.savefig(os.path.join(results_dir_, "time_evolution_test.pdf"))
    plt.close()

    # save data
    np.save(os.path.join(results_dir_, "gen_data"), data)
    np.save(os.path.join(results_dir_, "gen_data_test"), data_test)
    if generate_init_final:
        np.save(os.path.join(results_dir_, "gen_data_init"), data_init)
        np.save(os.path.join(results_dir_, "gen_data_final"), data_final)


if __name__ == "__main__":
    main()
