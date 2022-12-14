import pdb
import functools
import numpy as np

from utils import *


def init_Lorenz63_params():
    """
    Returns standard parameter configuration for L63 in chaotic regime.
    """
    params = {
        "rho": 28.0,
        "sigma": 10.0,
        "beta": 8.0 / 3.0,
    }
    return params


def final_Lorenz63_params():
    """
    Returns standard parameter configuration for L63 in chaotic regime.
    """
    params = {
        "rho": 19.6,
        "sigma": 7.0,
        "beta": 1.87,
    }
    return params


def init_Roessler_params():
    """
    Returns standard parameter configuration for L63 in chaotic regime.
    """
    params = {
        "a": 0.1,
        "b": 0.1,
        "c": 14.0,
    }
    return params


def final_Roessler_params():
    """
    Returns standard parameter configuration for L63 in chaotic regime.
    """
    params = {
        "a": 0.2,
        "b": 0.2,
        "c": 5.7,
    }
    return params


def init_bursting_neuron_params():
    """
    Returns standard parameter configuration for single neuron model in bursting regime.
    """
    params = {}
    params["I"] = 0  # mV
    params["C_m"] = 6  # μF
    params["g_L"] = 8  # mS
    params["E_L"] = -80  # mV
    params["g_Na"] = 20  # mS
    params["E_Na"] = 60  # mV
    params["V_h_Na"] = -20  # mV
    params["k_Na"] = 15
    params["g_K"] = 10  # mS
    params["E_K"] = -90  # mV
    params["V_h_K"] = -25  # mV
    params["k_K"] = 5
    params["tau_n"] = 1  # ms
    params["g_M"] = 25  # mS
    params["V_h_M"] = -15  # mV
    params["k_M"] = 5
    params["tau_h"] = 200  # ms  #200 or 35
    params["g_NMDA"] = 10.2  # mS
    params["E_NMDA"] = 0  # mV  #0
    return params


def get_init_params(system_name):
    if system_name == "Lorenz63":
        params = init_Lorenz63_params()
    elif system_name == "Bursting_Neuron":
        params = init_bursting_neuron_params()
    elif system_name == "Rössler":
        params = init_Roessler_params()
    else:
        raise NotImplementedError
    return params


def get_final_params(system_name):
    if system_name == "Lorenz63":
        params = final_Lorenz63_params()
    elif system_name == "Rössler":
        params = final_Roessler_params()
    else:
        params = NotImplemented
    return params


def get_params_from_schedule(params_init, schedules, t, step_size=None):
    """
    Compute the value of a non-stationary parameter at time t, according to a given schedule.
    """
    sched0 = list(schedules.values())[0]
    # if the schedule is an AR1 model
    if isinstance(sched0, functools.partial) and sched0.func.__name__ == "AR1_schedule":
        assert not step_size is None
        # for AR1 schedule, need to iterate an integer number of steps
        n = round(t / step_size)
        # initial parameters as array
        p = np.array(list(params_init.values()))
        new_p = sched0(p, n)
        params = dict(zip(params_init.keys(), new_p))
    # for any other schedule, apply compnonet-wise to each parameter
    else:
        params = {}
        for i, k in enumerate(params_init):
            param0 = params_init[k]
            schedule = schedules[k]
            params[k] = schedule(param0, t)
    return params


def get_params_all_time_steps(
    params_init: dict, params_final: dict, schedules: dict, T: int, step_size: int
):
    """
    Computes parameter values at all time steps given their respective schedules.

    Args:
        T (float) : System time, i.e. n_time_steps * step_size
        non_linearity (function) : function that describes non-linear time evolution of parameters
    """
    # dictionary that contains a full dictionary of all parameter values at each time step
    param_dict_all_time_steps = {}
    # dictionary to store all values for each parameter
    param_values = {}

    all_t = np.arange(0.0, T + 1, step_size)

    for i, k in enumerate(params_init):
        # calculate factor for linear schedule
        sched_k = schedules[k]
        if isinstance(sched_k, functools.partial) and sched_k.func.__name__ in [
            "exponential_schedule",
            "linear_schedule",
        ]:
            param_values[k] = sched_k(params_init[k], all_t)
        else:
            if sched_k.__name__ in ["oscillating_increase", "sigmoid"]:
                param_values[k] = transform_for_parameter(
                    all_t, sched_k, params_init[k], params_final[k]
                )
            elif sched_k.__name__ == "identity":
                param_values[k] = np.array([params_init[k]] * len(all_t))
            else:
                raise NotImplementedError(
                    f"Computation of parameter values at all times not available \
                                            for schedule {sched_k.__name__}"
                )

    for idx, t in enumerate(all_t):
        param_dict = {}
        for j, k in enumerate(params_init):
            param_dict[k] = param_values[k][idx]
        param_dict_all_time_steps[idx] = param_dict

    return param_dict_all_time_steps


def identity(params0, t):
    return params0


def schedules_all_stationary(params):
    """
    Returns dictionary with identity as schedule for all input parameters.
    """
    schedules = {}
    for i, param in enumerate(params):
        schedules[param] = identity
    return schedules


###################
# linear schedules
###################


def linear_schedule(param0: float, t: int, alpha: float):
    return param0 + t * alpha


def get_linear_schedule(alpha):
    return functools.partial(linear_schedule, alpha=alpha)


def AR1_schedule(param0: float, t: int, R0, R1):
    assert param0.shape == R0.shape
    return iter_linear_map(R1, R0, param0, t)


def get_AR1_schedule(R0, R1):
    return functools.partial(AR1_schedule, R0=R0, R1=R1)


def get_linear_schedule_from_final_param(
    param_init: float, param_final: float, T: float
):
    alpha_k = (param_final - param_init) / T
    schedule_k = get_linear_schedule(alpha_k)
    return schedule_k


def get_AR1_schedule_for_params(params_init: dict, R0, R1):
    sched = get_AR1_schedule(R0, R1)
    schedules = {}
    for i, k in enumerate(params_init):
        schedules[k] = sched
    return schedules


def get_linear_schedules_from_final_params(
    params_init: dict, params_final: dict, T: float
):
    """
    Given final and initial parameters, return factor for linear schedule
    so that: schedule(param_init, T) = param_final.

    Args:
        T (float) : System time, i.e. n_time_steps * step_size
    """
    assert params_init.keys() == params_final.keys()
    schedules = {}
    for i, k in enumerate(params_init):
        # calculate factor for linear schedule
        alpha_k = (params_final[k] - params_init[k]) / T
        schedules[k] = get_linear_schedule(alpha_k)
    return schedules


#######################
# non-linear schedules
#######################


def exponential_schedule(
    param0: float, t: int, alpha: float, tau: float, t_offset: float = 0.0
):
    return param0 + alpha * np.exp((t + t_offset) / tau)


def get_exponential_schedule(alpha, tau, t_offset=0.0):
    return functools.partial(
        exponential_schedule, alpha=alpha, tau=tau, t_offset=t_offset
    )


def oscillating_increase(x, t):
    """
    Linear increase, modulated by sin function.
    """
    assert isinstance(x, np.ndarray)
    period = 2 / len(x)
    out = 1.0 * x + 0.6 * x.mean() * np.sin(x * period * 2 * np.pi)
    return out


def sigmoid(x, t):
    """
    Sigmoid function.
    """
    assert isinstance(x, np.ndarray)
    out = 1 / (1 + np.exp(-25 / len(x) * (x - len(x) / 2)))
    return out


def transform_for_parameter(all_t, func, init, final):
    """
    Transforms function so that it has desired initial and final values.
    """
    xs = np.arange(len(all_t))
    ys = func(xs, all_t)
    # if function is identity map
    # if np.array_equal(xs, ys):
    #    return xs
    min_ = init - ys[0]
    slope = (final - init) / (ys[-1] - ys[0])
    out = ys * slope + min_
    return out
