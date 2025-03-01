# -----------------------------------------------------------------------------
# Copyright (c) 2025 Shaw Wang from XJTU, ShawWang@yeah.net
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

# referenced from https://msp.org/pjm/1990/145-2/pjm-v145-n2-p06-s.pdf
# referenced from https://ompl.kavrakilab.org/ReedsSheppStateSpace_8cpp_source.html
# referenced from https://blog.csdn.net/qq_44339029/article/details/126200191
# referenced from https://blog.csdn.net/jianmo1993/article/details/143200673
# referenced from https://cpb-us-e2.wpmucdn.com/faculty.sites.uci.edu/dist/e/700/files/2014/04/Dubins_Set_Robotics_2001.pdf
# referenced from https://blog.csdn.net/qq_44339029/article/details/126095951


import time
from functools import wraps

CAR_PATH_MODULE_TIMER_ENABLED = False


def timer(enabled=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled:
                start_time = time.time()  # 记录开始时间
                result = func(*args, **kwargs)
                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time
                print(
                    f"Function '{func.__name__}' executed in {elapsed_time:.9f} seconds"
                )
            else:
                result = func(*args, **kwargs)  # 如果不计时，直接执行
            return result

        return wrapper

    return decorator


import numpy as np

_Pi_2 = 0.5 * np.pi
_Pi = 1.0 * np.pi
_2_Pi = 2.0 * np.pi


def turn_motion(x, y, phi, kappa, delta_phi):
    """
    params:
        x: x-coordinate of the current position
        y: y-coordinate of the current position
        phi: current heading angle
        kappa: curvature, positive for left turn, negative for right turn
        delta_phi: angle of turn,
                positive for left forward turn, and right backward turn,
                negative for right forward turn and left backward turn
    returns:
        x_new: x-coordinate of the new position
        y_new: y-coordinate of the new position
        phi_new: new heading angle
    """
    phi_new = (phi + delta_phi) % _2_Pi
    x_new = x + (np.sin(phi_new) - np.sin(phi)) / kappa
    y_new = y - (np.cos(phi_new) - np.cos(phi)) / kappa
    return x_new, y_new, phi_new


def line_motion(x, y, phi, delta_s):
    """
    params:
        x: x-coordinate of the current position
        y: y-coordinate of the current position
        phi: current heading angle
        delta_s: distance to move, positive for forward, negative for backward
    returns:
        x_new: x-coordinate of the new position
        y_new: y-coordinate of the new position
        phi_new: new heading angle
    """
    x_new = x + np.cos(phi) * delta_s
    y_new = y + np.sin(phi) * delta_s
    phi_new = np.ones_like(delta_s) * phi
    return x_new, y_new, phi_new


@timer(CAR_PATH_MODULE_TIMER_ENABLED)
def path_interp(types, lens, step, kappa, x=0, y=0, phi=0):
    """
    params:
        types: types of the path segments
        lens: lengths of the path segments
        step: step size of the path interpolation
        kappa: curvature of reeds-shepp path
        x: x-coordinate of the start position
        y: y-coordinate of the start position
        phi: start heading angle
    returns:
        path_x: local x-coordinate of the rs path
        path_y: local y-coordinate of the rs path
        path_phi: local heading angle of the rs path
    """
    step = step * kappa
    path_x = []
    path_y = []
    path_phi = []
    # 计算每段路径的插值点
    for type, len in zip(types, lens):
        steps = np.arange(0, len, step)
        if type == "Lp":
            seg_x, seg_y, seg_phi = turn_motion(x, y, phi, kappa, steps)
            x, y, phi = turn_motion(x, y, phi, kappa, len)
        elif type == "Sp":
            seg_x, seg_y, seg_phi = line_motion(x, y, phi, steps / kappa)
            x, y, phi = line_motion(x, y, phi, len / kappa)
        elif type == "Rp":
            seg_x, seg_y, seg_phi = turn_motion(x, y, phi, -kappa, -steps)
            x, y, phi = turn_motion(x, y, phi, -kappa, -len)
        elif type == "Rm":
            seg_x, seg_y, seg_phi = turn_motion(x, y, phi, -kappa, steps)
            x, y, phi = turn_motion(x, y, phi, -kappa, len)
        elif type == "Sm":
            seg_x, seg_y, seg_phi = line_motion(x, y, phi, -steps / kappa)
            x, y, phi = line_motion(x, y, phi, -len / kappa)
        elif type == "Lm":
            seg_x, seg_y, seg_phi = turn_motion(x, y, phi, kappa, -steps)
            x, y, phi = turn_motion(x, y, phi, kappa, -len)
        path_x.append(seg_x)
        path_y.append(seg_y)
        path_phi.append(seg_phi)
    path_x.append([x])
    path_y.append([y])
    path_phi.append([phi])
    path_x = np.concatenate(path_x)
    path_y = np.concatenate(path_y)
    path_phi = np.concatenate(path_phi)
    return path_x, path_y, path_phi


def _corrd_trans(x, y, phi, x_, y_, phi_, kappa):
    """
    params:
        x: x-coordinate of the current position
        y: y-coordinate of the current position
        phi: current heading angle
        x_: x-coordinate of the target position
        y_: y-coordinate of the target position
        phi_: target heading angle
        kappa: curvature of reeds-shepp path
    returns:
        delta_x: x-coordinate of the local corrdinate
        delta_y: y-coordinate of the local corrdinate
        delta_phi: heading angle of the local corrdinate
    """
    delta_x = (x_ - x) * kappa
    delta_y = (y_ - y) * kappa
    delta = np.array([delta_x, delta_y])
    rot_mat = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    delta = rot_mat @ delta
    delta_phi = (phi_ - phi) % _2_Pi
    return *delta, delta_phi


def _polar_trans(x, y):
    """
    params:
        x: x-coordinate of the current position
        y: y-coordinate of the current position
    returns:
        rh: distance from the origin
        theta: angle from the positive x-axis
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta % _2_Pi


# dubins path


def _dubins_LpSpLp(x, y, phi):
    """
    x = u*cos(t) + sin(t + v)
    y = u*sin(t) - cos(t + v) + 1
    phi = t + v
    """
    u, t = _polar_trans(x - np.sin(phi), y + np.cos(phi) - 1)
    t = t % _2_Pi
    v = (phi - t) % _2_Pi
    return True, [t, u, v]


def _dubins_RpSpRp(x, y, phi):
    """
    x = u*cos(t) + sin(t + v)
    y = -u*sin(t) + cos(t + v) - 1
    phi = -t - v
    """
    u, theta = _polar_trans(y + 1 - np.cos(phi), x + np.sin(phi))
    t = (theta - _Pi_2) % _2_Pi
    v = (-t - phi) % _2_Pi
    return True, [t, u, v]


def _dubins_LpSpRp(x, y, phi):
    """
    x = u*cos(t) + 2*sin(t) - sin(t - v)
    y = u*sin(t) - 2*cos(t) + cos(t - v) + 1
    phi = t - v
    """
    u1, t1 = _polar_trans(x + np.sin(phi), y - np.cos(phi) - 1)
    if u1 >= 2:
        u = np.sqrt(u1**2 - 4)
        t = (t1 + np.arctan2(2, u)) % _2_Pi
        v = (t - phi) % _2_Pi
        return True, [t, u, v]
    return False, []


def dubins_RpSpLp(x, y, phi):
    """
    x = u*cos(t) + 2*sin(t) - sin(t - v)
    y = -u*sin(t) + 2*cos(t) - cos(t - v) - 1
    phi = -t + v
    """
    u1, t1 = _polar_trans(y + 1 + np.cos(phi), x - np.sin(phi))
    if u1 >= 2:
        u = np.sqrt(u1**2 - 4)
        t = (t1 - np.arctan2(u, 2)) % _2_Pi
        v = (phi + t) % _2_Pi
        return True, [t, u, v]
    return False, []


def _dubins_LpRpLp(x, y, phi):
    """
    x = 2*sin(t) - 2*sin(t - u) + sin(t - u + v)
    y = -2*cos(t) + 2*cos(t - u) - cos(t - u + v) + 1
    phi = t - u + v
    """
    u1, t1 = _polar_trans(x - np.sin(phi), y + np.cos(phi) - 1)
    if u1 <= 4:
        u = 2 * np.arcsin(u1 / 4)
        t = (t1 + u / 2) % _2_Pi
        v = (phi + u - t) % _2_Pi
        return True, [t, u, v]
    return False, []


def _dubins_RpLpRp(x, y, phi):
    """
    x = 2*sin(t) - 2*sin(t - u) + sin(t - u + v)
    y = 2*cos(t) - 2*cos(t - u) + cos(t - u + v) - 1
    phi = -t + u - v
    """
    u1, t1 = _polar_trans(y - np.cos(phi) + 1, x + np.sin(phi))
    if u1 <= 4:
        u = 2 * np.arcsin(u1 / 4)
        t = (t1 + u / 2 - _Pi_2) % _2_Pi
        v = (u - t - phi) % _2_Pi
        return True, [t, u, v]
    return False, []


def _dubins_CSC(x, y, phi, sols):
    flag, lens = _dubins_LpSpLp(x, y, phi)
    if flag:
        sols.append({"cost": np.sum(lens), "type": ["Lp", "Sp", "Lp"], "lens": lens})
    flag, lens = _dubins_RpSpRp(x, y, phi)
    if flag:
        sols.append({"cost": np.sum(lens), "type": ["Rp", "Sp", "Rp"], "lens": lens})
    flag, lens = _dubins_LpSpRp(x, y, phi)
    if flag:
        sols.append({"cost": np.sum(lens), "type": ["Lp", "Sp", "Rp"], "lens": lens})
    flag, lens = dubins_RpSpLp(x, y, phi)
    if flag:
        sols.append({"cost": np.sum(lens), "type": ["Rp", "Sp", "Lp"], "lens": lens})
    return sols


def _dubins_CCC(x, y, phi, sols):
    flag, lens = _dubins_LpRpLp(x, y, phi)
    if flag:
        sols.append({"cost": np.sum(lens), "type": ["Lp", "Rp", "Lp"], "lens": lens})
    flag, lens = _dubins_RpLpRp(x, y, phi)
    if flag:
        sols.append({"cost": np.sum(lens), "type": ["Rp", "Lp", "Rp"], "lens": lens})
    return sols


@timer(CAR_PATH_MODULE_TIMER_ENABLED)
def dubins_all_paths(x, y, phi, x_, y_, phi_, kappa):
    """
    params:
        x: x-coordinate of the current position
        y: y-coordinate of the current position
        phi: current heading angle
        x_: x-coordinate of the target position
        y_: y-coordinate of the target position
        phi_: target heading angle
        kappa: curvature of path
    returns:
        dubins_paths: a list of dubins paths, each path is a dictionary with keys "cost", "type", "lens"
    """
    sols = []
    loc = _corrd_trans(x, y, phi, x_, y_, phi_, kappa)
    _dubins_CSC(*loc, sols)
    _dubins_CCC(*loc, sols)
    return sols


@timer(CAR_PATH_MODULE_TIMER_ENABLED)
def dubins_shortest_path(x, y, phi, x_, y_, phi_, kappa):
    return min(
        dubins_all_paths(x, y, phi, x_, y_, phi_, kappa), key=lambda sol: sol["cost"]
    )


@timer(CAR_PATH_MODULE_TIMER_ENABLED)
def dubins_shortest_path_length(x, y, phi, x_, y_, phi_, kappa):
    return dubins_shortest_path(x, y, phi, x_, y_, phi_, kappa)["cost"]


# reeds-shepp path


def _LpSpLp(x, y, phi):
    """
    init: [0, 0, 0], motion: ['L+', 'S+', 'L+'], lens: [t, u, v]
    x = u*cos(t) + sin(t + v) -> u*cos(t) = x - sin(phi)
    y = u*sin(t) - cos(t + v) + 1 -> u*sin(t) = y + cos(phi) - 1
    phi = t + v
    u, t = polar(x - sin(phi), y + cos(phi) - 1)
        -> v = phi -t
    """
    u, t = _polar_trans(x - np.sin(phi), y + np.cos(phi) - 1)
    v = (phi - t) % _2_Pi
    if t <= _Pi and v <= _Pi:
        return True, t, u, v
    return False, None, None, None


def _LpSpRp(x, y, phi):
    """
    init: [0, 0, 0], motion: ['L+', 'S+', 'R+'], lens: [t, u, v]
    x = u*cos(t) + 2*sin(t) - sin(t - v)
    y = u*sin(t) - 2*cos(t) + cos(t - v) + 1 -> u*sin(t) = y - cos(phi) - 1
    phi = t - v
        -> u*cos(t) + 2*cos(t - pi/2) = x + sin(phi)
        -> u*sin(t) + 2*sin(t - pi/2) = y - cos(phi) - 1
    u1, t1 = polar(x + np.sin(phi), y - np.cos(phi) - 1)
        -> u = sqrt(u1**2 - 4)
        -> t = t1 + atan(2/u1)
        -> v = t - phi
    """
    u1, t1 = _polar_trans(x + np.sin(phi), y - np.cos(phi) - 1)
    if u1 > 2.0:
        u = np.sqrt(u1**2 - 4)
        beta = np.arctan2(2, u)
        t = (t1 + beta) % _2_Pi
        v = (t - phi) % _2_Pi
        if t <= _Pi and v <= _Pi:
            return True, t, u, v
    return False, None, None, None


def _CSC(x, y, phi, sols):
    flag, t, u, v = _LpSpLp(x, y, phi)
    if flag:  # LpSpLp
        sols.append({"type": ["Lp", "Sp", "Lp"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpSpLp(-x, y, -phi)  # timeflip
    if flag:  # LmSmLm
        sols.append({"type": ["Lm", "Sm", "Lm"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpSpLp(x, -y, -phi)  # reflect
    if flag:  # RpSpRp
        sols.append({"type": ["Rp", "Sp", "Rp"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpSpLp(-x, -y, phi)  # timeflip + reflect
    if flag:  # RmSmRm
        sols.append({"type": ["Rm", "Sm", "Rm"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpSpRp(x, y, phi)
    if flag:  # LpSpRp
        sols.append({"type": ["Lp", "Sp", "Rp"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpSpRp(-x, y, -phi)  # timeflip
    if flag:  # LmSmRm
        sols.append({"type": ["Lm", "Sm", "Rm"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpSpRp(x, -y, -phi)  # reflect
    if flag:  # RpSpLp
        sols.append({"type": ["Rp", "Sp", "Lp"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpSpRp(-x, -y, phi)  # timeflip + reflect
    if flag:  # RmSmLm
        sols.append({"type": ["Rm", "Sm", "Lm"], "lens": [t, u, v], "cost": t + u + v})


def _LpRmLp(x, y, phi):
    """
    init: [0, 0, 0], motion: ['L+', 'R-', 'L+'], lens: [t, u, v]
    x = 2*sin(t) - 2*sin(t + u) + sin(t + u + v)
    y = -2*cos(t) + 2*cos(t + u) - cos(t + u + v) + 1
    phi = t + u + v
        -> 2*sin(t+u) - 2*sin(t) = -(x - sin(phi))
            -> 4*cos(u/2)*cos(t + u/2) = -(x - sin(phi))
        -> 2*cos(t+u) - 2*cos(t) = y + cos(phi) - 1
            -> 4*sin(u/2)*sin(t + u/2) = -(y + cos(phi) - 1)
    u1, t1 = polar(-(x - sin(phi)), -(y + cos(phi) - 1))
        -> u = 2*arcsin(u1/4)
        -> t = theta - u/2
        -> v = phi - t - u
    """
    u1, t1 = _polar_trans(-(x - np.sin(phi)), -(y + np.cos(phi) - 1))
    if u1 <= 4.0:
        u = 2.0 * np.arcsin(0.25 * u1)
        t = (t1 - 0.5 * u) % _2_Pi
        v = (phi - t - u) % _2_Pi
        if t <= _Pi and u <= _Pi and v <= _Pi:
            return True, t, u, v
    return False, None, None, None


def _LpRmLm(x, y, phi):
    """
    init: [0, 0, 0], motion: ['L+', 'R-', 'L-'], lens: [t, u, v]
    x = 2*sin(t) - 2*sin(t + u) + sin(t + u - v)
    y = -2*cos(t) + 2*cos(t + u) - cos(t + u - v) + 1
    phi = t + u - v
        -> 2*sin(t+u) - 2*sin(t) = x - sin(phi)
            -> 4*cos(u/2)*cos(t + u/2) = -(x - sin(phi))
        -> 2*cos(t+u) - 2*cos(t) = y + cos(phi) - 1
            -> 4*sin(u/2)*sin(t + u/2) = -(y + cos(phi) - 1)
    u1, t1 = polar(-(x - np.sin(phi)), -(y + np.cos(phi) - 1))
        -> u = 2*arcsin(u1/4)
        -> t = theta - u/2
        -> v = t + u - phi
    """
    u1, theta = _polar_trans(-(x - np.sin(phi)), -(y + np.cos(phi) - 1))
    if u1 <= 4.0:
        u = 2.0 * np.arcsin(0.25 * u1)
        t = (theta - 0.5 * u) % _2_Pi
        v = (t + u - phi) % _2_Pi
        if t <= _Pi and u <= _Pi and v <= _Pi:
            return True, t, u, v
    return False, None, None, None


def _CCC(x, y, phi, sols):
    flag, t, u, v = _LpRmLp(x, y, phi)
    if flag:  # LpRmLp
        sols.append({"type": ["Lp", "Rm", "Lp"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpRmLp(-x, y, -phi)  # timeflip
    if flag:  # LmRpLm
        sols.append({"type": ["Lm", "Rp", "Lm"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpRmLp(x, -y, -phi)  # reflect
    if flag:  # RpLmRp
        sols.append({"type": ["Rp", "Lm", "Rp"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpRmLp(-x, -y, phi)  # timeflip + reflect
    if flag:  # RmLpRm
        sols.append({"type": ["Rm", "Lp", "Rm"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpRmLm(x, y, phi)
    if flag:  # LpRmLm
        sols.append({"type": ["Lp", "Rm", "Lm"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpRmLm(-x, y, -phi)  # timeflip
    if flag:  # LmRpLp
        sols.append({"type": ["Lm", "Rp", "Lp"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpRmLm(x, -y, -phi)  # reflect
    if flag:  # RpLmRm
        sols.append({"type": ["Rp", "Lm", "Rm"], "lens": [t, u, v], "cost": t + u + v})
    flag, t, u, v = _LpRmLm(-x, -y, phi)  # timeflip + reflect
    if flag:  # RmLpRp
        sols.append({"type": ["Rm", "Lp", "Rp"], "lens": [t, u, v], "cost": t + u + v})
    # backwards
    xb = x * np.cos(phi) + y * np.sin(phi)
    yb = x * np.sin(phi) - y * np.cos(phi)
    flag, t, u, v = _LpRmLm(xb, yb, phi)
    if flag:  # LmRmLp
        sols.append({"type": ["Lm", "Rm", "Lp"], "lens": [v, u, t], "cost": t + u + v})
    flag, t, u, v = _LpRmLm(-xb, yb, -phi)  # timeflip
    if flag:  # LpRpLm
        sols.append({"type": ["Lp", "Rp", "Lm"], "lens": [v, u, t], "cost": t + u + v})
    flag, t, u, v = _LpRmLm(xb, -yb, -phi)  # reflect
    if flag:  # RmLmRp
        sols.append({"type": ["Rm", "Lm", "Rp"], "lens": [v, u, t], "cost": t + u + v})
    flag, t, u, v = _LpRmLm(-xb, -yb, phi)  # timeflip + reflect
    if flag:  # RpLpRm
        sols.append({"type": ["Rp", "Lp", "Rm"], "lens": [v, u, t], "cost": t + u + v})


def _LpRupLumRm(x, y, phi):
    """
    init: [0, 0, 0], motion: ['L+', 'R+', 'L-', 'R-'], lens: [t, u, u, v]
    x = 2*sin(t) + 2*sin(t - 2*u) - 2*sin(t - u) - sin(t - 2*u + v)
    y = -2*cos(t) - 2*cos(t - 2*u) + 2*cos(t - u) + cos(t - 2*u + v) + 1
    phi = t - 2*u + v
        -> 2*sin(t-2*u) - 2*sin(t-u) + 2*sin(t) = x + sin(phi)
        -> -2*cos(t-2*u) + 2*cos(t-u) - 2*cos(t) = y - cos(phi) - 1
            -> 4*cos(u)*sin(t - u) - 2*sin(t - u) = x + sin(phi)
            -> -4*cos(u)*cos(t-u) + 2*cos(t-u) = y - cos(phi) - 1
                -> (4*cos(u) - 2)*cos(t - u - pi/2) = x + sin(phi)
                -> (4*cos(u) - 2)*sin(t - u - pi/2) = y - cos(phi) - 1
    u1, t1 = polar(x + np.sin(phi), y - np.cos(phi) - 1)
        -> u = arccos((u1+2)/4)
        -> t = theta + u + pi/2
        -> v = phi + 2*u - t
    """
    u1, t1 = _polar_trans(x + np.sin(phi), y - np.cos(phi) - 1)
    if u1 <= 2.0:
        u = np.arccos((u1 + 2.0) / 4.0)
        t = (t1 + u + _Pi_2) % _2_Pi
        v = (phi + 2 * u - t) % _2_Pi
        if t <= _Pi and v <= _Pi:
            return True, t, u, v
    return False, None, None, None


def _LpRumLumRp(x, y, phi):
    """
    init: [0, 0, 0], motion: ['L+', 'R-', 'L-', 'R+'], lens: [t, u, u, v]
    x = 4*sin(t) - 2*sin(t + u) - sin(t - v)
    y = -4*cos(t) + 2*cos(t + u) + cos(t - v) + 1
    phi = t - v
        -> 4*sin(t) - 2*sin(t + u) = x + sin(phi)
        -> -4*cos(t) + 2*cos(t + u) = y - cos(phi) - 1
            -> 4*cos(t-pi/2) + 2*cos(t+u+pi/2) = x + sin(phi)
            -> 4*sin(t-pi/2) + 2*sin(t+u+pi/2) = y - cos(phi) - 1
    u1, t1 = polar(x + np.sin(phi), y - np.cos(phi) - 1)
    作图分析，可以得到：
        cos(u) = (4^2+2^2-u1^2)/(2*4*2)
        t - pi/2 - beta = t1, 其中beta = arcsin((2*sin(u1)/u1)
        v = t - phi
    """
    u1, theta = _polar_trans(x + np.sin(phi), y - np.cos(phi) - 1)
    u2 = 20 - u1**2
    if 0 <= u2 <= 16:
        u = np.arccos(u2 / 16)  # [0, pi]
        t = (theta + np.arcsin((2 * np.sin(u)) / u1) + _Pi / 2) % _2_Pi
        v = (t - phi) % _2_Pi
        if t <= _Pi and v <= _Pi:
            return True, t, u, v
    return False, None, None, None


def _CCCC(x, y, phi, sols):
    flag, t, u, v = _LpRupLumRm(x, y, phi)
    if flag:  # LpRupLumRm
        sols.append(
            {
                "type": ["Lp", "Rp", "Lm", "Rm"],
                "lens": [t, u, u, v],
                "cost": t + u + u + v,
            }
        )
    flag, t, u, v = _LpRupLumRm(-x, y, -phi)  # timeflip
    if flag:  # LmRumLupRp
        sols.append(
            {
                "type": ["Lm", "Rm", "Lp", "Rp"],
                "lens": [t, u, u, v],
                "cost": t + u + u + v,
            }
        )
    flag, t, u, v = _LpRupLumRm(x, -y, -phi)  # reflect
    if flag:  # RpLupRumLm
        sols.append(
            {
                "type": ["Rp", "Lp", "Rm", "Lm"],
                "lens": [t, u, u, v],
                "cost": t + u + u + v,
            }
        )
    flag, t, u, v = _LpRupLumRm(-x, -y, phi)  # timeflip + reflect
    if flag:  # RmLumRupLp
        sols.append(
            {
                "type": ["Rm", "Lm", "Rp", "Lp"],
                "lens": [t, u, u, v],
                "cost": t + u + u + v,
            }
        )
    flag, t, u, v = _LpRumLumRp(x, y, phi)
    if flag:  # LpRumLumRp
        sols.append(
            {
                "type": ["Lp", "Rm", "Lm", "Rp"],
                "lens": [t, u, u, v],
                "cost": t + u + u + v,
            }
        )
    flag, t, u, v = _LpRumLumRp(-x, y, -phi)  # timeflip
    if flag:  # LmRupLupRm
        sols.append(
            {
                "type": ["Lm", "Rp", "Lp", "Rm"],
                "lens": [t, u, u, v],
                "cost": t + u + u + v,
            }
        )
    flag, t, u, v = _LpRumLumRp(x, -y, -phi)  # reflect
    if flag:  # RpLumRumLp
        sols.append(
            {
                "type": ["Rp", "Lm", "Rm", "Lp"],
                "lens": [t, u, u, v],
                "cost": t + u + u + v,
            }
        )
    flag, t, u, v = _LpRumLumRp(-x, -y, phi)  # timeflip + reflect
    if flag:  # RmLupRupLm
        sols.append(
            {
                "type": ["Rm", "Lp", "Rp", "Lm"],
                "lens": [t, u, u, v],
                "cost": t + u + u + v,
            }
        )


def _LpRmSmRm(x, y, phi):
    """
    init: [0, 0, 0], motion: ['L+', 'R-', 'S-', 'R-'], lens: [t, pi_2, u, v]
    x = -u*cos(pi_2 + t) + 2*sin(t) - sin(pi_2 + t + v)
    y = -u*sin(pi_2 + t) - 2*cos(t) + cos(pi_2 + t + v) + 1
    phi = pi_2 + t + v
        -> -u*cos(pi_2 + t) + 2*sin(t) = x + sin(phi)
        -> -u*sin(pi_2 + t) - 2*cos(t) = y - cos(phi) - 1
            -> u*cos(t - pi_2) + 2*cos(t - pi_2) = x + sin(phi)
            -> u*sin(t - pi_2) + 2*sin(t - pi_2) = y - cos(phi) - 1
                ->(u + 2)*cos(t - pi_2) = x + sin(phi)
                ->(u + 2)*sin(t - pi_2) = y - cos(phi) - 1
    u1, t1 = polar(x + np.sin(phi), y - np.cos(phi) - 1)
        -> u = u1 - 2
        -> t = t1 + pi_2
        -> v = phi - t - pi_2
    """
    u1, t1 = _polar_trans(x + np.sin(phi), y - np.cos(phi) - 1)
    if u1 >= 2.0:
        u = u1 - 2.0
        t = (t1 + _Pi_2) % _2_Pi
        v = (phi - t - _Pi_2) % _2_Pi
        if t <= _Pi and v <= _Pi:
            return True, t, u, v
    return False, None, None, None


def _LpRmSmLm(x, y, phi):
    """
    init: [0, 0, 0], motion: ['L+', 'R-', 'S-', 'L-'], lens: [t, pi_2, u, v]
    x = -u*cos(pi_2 + t) + 2*sin(t) - 2*sin(pi_2 + t) + sin(pi_2 + t - v)
    y = -u*sin(pi_2 + t) - 2*cos(t) + 2*cos(pi_2 + t) - cos(pi_2 + t - v) + 1
    phi = pi_2 + t - v
        -> -u*cos(pi_2 + t) + 2*sin(t) - 2*sin(pi_2 + t) = x - sin(phi)
        -> -u*sin(pi_2 + t) - 2*cos(t) + 2*cos(pi_2 + t) = y + cos(phi) - 1
            -> u*cos(t - pi_2) + 2*cos(t - pi_2) - 2*cos(t) = x - sin(phi)
            -> u*sin(t - pi_2) + 2*sin(t - pi_2) - 2*sin(t) = y + cos(phi) - 1
                ->(u + 2)*cos(t - pi_2) - 2*cos(t) = x - sin(phi)
                ->(u + 2)*sin(t - pi_2) - 2*sin(t) = y + cos(phi) - 1
    u1, t1 = polar(x - np.sin(phi), y + np.cos(phi) - 1)
    作图分析，可以得到：
        u = sqrt(u1^2 - 4) - 2
        t = t1 + arctan(2/(u+2)) + pi/2
        v = t - phi + pi/2
    """
    u1, t1 = _polar_trans(x - np.sin(phi), y + np.cos(phi) - 1)
    if u1 >= 2.0 * np.sqrt(2.0):
        u = np.sqrt(u1**2 - 4.0) - 2.0
        t = (t1 + np.arctan(2.0 / (u + 2.0)) + _Pi_2) % _2_Pi
        v = (t - phi + _Pi_2) % _2_Pi
        if t <= _Pi and v <= _Pi:
            return True, t, u, v
    return False, None, None, None


def _CCSC(x, y, phi, sols):
    flag, t, u, v = _LpRmSmRm(x, y, phi)
    if flag:  # LpRmSmRm
        sols.append(
            {
                "type": ["Lp", "Rm", "Sm", "Rm"],
                "lens": [t, _Pi_2, u, v],
                "cost": t + _Pi_2 + u + v,
            }
        )
    flag, t, u, v = _LpRmSmRm(-x, y, -phi)  # timeflip
    if flag:  # LmRpSpRp
        sols.append(
            {
                "type": ["Lm", "Rp", "Sp", "Rp"],
                "lens": [t, _Pi_2, u, v],
                "cost": t + _Pi_2 + u + v,
            }
        )
    flag, t, u, v = _LpRmSmRm(x, -y, -phi)  # reflect
    if flag:  # RpLmSmLm
        sols.append(
            {
                "type": ["Rp", "Lm", "Sm", "Lm"],
                "lens": [t, _Pi_2, u, v],
                "cost": t + _Pi_2 + u + v,
            }
        )
    flag, t, u, v = _LpRmSmRm(-x, -y, phi)  # timeflip + reflect
    if flag:  # RmLpSpLp
        sols.append(
            {
                "type": ["Rm", "Lp", "Sp", "Lp"],
                "lens": [t, _Pi_2, u, v],
                "cost": t + _Pi_2 + u + v,
            }
        )
    flag, t, u, v = _LpRmSmLm(x, y, phi)
    if flag:  # LpRmSmLm
        sols.append(
            {
                "type": ["Lp", "Rm", "Sm", "Lm"],
                "lens": [t, _Pi_2, u, v],
                "cost": t + _Pi_2 + u + v,
            }
        )
    flag, t, u, v = _LpRmSmLm(-x, y, -phi)  # timeflip
    if flag:  # LmRpSpLp
        sols.append(
            {
                "type": ["Lm", "Rp", "Sp", "Lp"],
                "lens": [t, _Pi_2, u, v],
                "cost": t + _Pi_2 + u + v,
            }
        )
    flag, t, u, v = _LpRmSmLm(x, -y, -phi)  # reflect
    if flag:  # RpLmSmRm
        sols.append(
            {
                "type": ["Rp", "Lm", "Sm", "Rm"],
                "lens": [t, _Pi_2, u, v],
                "cost": t + _Pi_2 + u + v,
            }
        )
    flag, t, u, v = _LpRmSmLm(-x, -y, phi)  # timeflip + reflect
    if flag:  # RmLpSpRp
        sols.append(
            {
                "type": ["Rm", "Lp", "Sp", "Rp"],
                "lens": [t, _Pi_2, u, v],
                "cost": t + _Pi_2 + u + v,
            }
        )


def _CSCC(x, y, phi, sols):
    xb = x * np.cos(phi) + y * np.sin(phi)
    yb = x * np.sin(phi) - y * np.cos(phi)
    flag, t, u, v = _LpRmSmRm(xb, yb, phi)
    if flag:  # RmSmRmLp
        sols.append(
            {
                "type": ["Rm", "Sm", "Rm", "Lp"],
                "lens": [v, u, _Pi_2, t],
                "cost": t + u + _Pi_2 + v,
            }
        )
    flag, t, u, v = _LpRmSmRm(-xb, yb, -phi)  # timeflip
    if flag:  # RpSpRpLm
        sols.append(
            {
                "type": ["Rp", "Sp", "Rp", "Lm"],
                "lens": [v, u, _Pi_2, t],
                "cost": t + u + _Pi_2 + v,
            }
        )
    flag, t, u, v = _LpRmSmRm(xb, -yb, -phi)  # reflect
    if flag:  # LmSmLmRp
        sols.append(
            {
                "type": ["Lm", "Sm", "Lm", "Rp"],
                "lens": [v, u, _Pi_2, t],
                "cost": t + u + _Pi_2 + v,
            }
        )
    flag, t, u, v = _LpRmSmRm(-xb, -yb, phi)  # timeflip + reflect
    if flag:  # LpSpLpRm
        sols.append(
            {
                "type": ["Lp", "Sp", "Lp", "Rm"],
                "lens": [v, u, _Pi_2, t],
                "cost": t + u + _Pi_2 + v,
            }
        )
    flag, t, u, v = _LpRmSmLm(xb, yb, phi)
    if flag:  # LmSmRmLp
        sols.append(
            {
                "type": ["Lm", "Sm", "Rm", "Lp"],
                "lens": [v, u, _Pi_2, t],
                "cost": t + u + _Pi_2 + v,
            }
        )
    flag, t, u, v = _LpRmSmLm(-xb, yb, -phi)  # timeflip
    if flag:  # LpSpRpLm
        sols.append(
            {
                "type": ["Lp", "Sp", "Rp", "Lm"],
                "lens": [v, u, _Pi_2, t],
                "cost": t + u + _Pi_2 + v,
            }
        )
    flag, t, u, v = _LpRmSmLm(xb, -yb, -phi)  # reflect
    if flag:  # RmSmLmRp
        sols.append(
            {
                "type": ["Rm", "Sm", "Lm", "Rp"],
                "lens": [v, u, _Pi_2, t],
                "cost": t + u + _Pi_2 + v,
            }
        )
    flag, t, u, v = _LpRmSmLm(-xb, -yb, phi)  # timeflip + reflect
    if flag:  # RpSpLpRm
        sols.append(
            {
                "type": ["Rp", "Sp", "Lp", "Rm"],
                "lens": [v, u, _Pi_2, t],
                "cost": t + u + _Pi_2 + v,
            }
        )


def _LpRmSmLmRp(x, y, phi):
    """
    init: [0, 0, 0], motion: ['L+', 'R-', 'S-', 'L-', 'R+'], lens: [t, pi_2, u, pi_2, v]
    x = -u*cos(pi_2 + t) + 4*sin(t) - 2*sin(pi_2 + t) - sin(t - v)
    y = -u*sin(pi_2 + t) - 4*cos(t) + 2*cos(pi_2 + t) + cos(t - v) + 1
    phi = t - v
        -> u*cos(t - pi_2) + 4*cos(t - pi_2) - 2*cos(t) = x + sin(phi)
        -> u*sin(t - pi_2) + 4*sin(t - pi_2) - 2*sin(t) = y - cos(phi) - 1
            ->(u + 4)*cos(t - pi_2) - 2*cos(t) = x + sin(phi)
            ->(u + 4)*sin(t - pi_2) - 2*sin(t) = y - cos(phi) - 1
    u1, t1 = polar(x + np.sin(phi), y - np.cos(phi) - 1)
    作图分析，可以得到：
        u = sqrt(u1^2 - 4) - 4
        t = t1 + arctan(2/(u+4)) + pi/2
        v = t - phi
    """
    u1, t1 = _polar_trans(x + np.sin(phi), y - np.cos(phi) - 1)
    if u1 >= 2 * np.sqrt(5.0):
        u = np.sqrt(u1**2 - 4.0) - 4.0
        t = (t1 + np.arctan(2.0 / (u + 4.0)) + _Pi_2) % _2_Pi
        v = (t - phi) % _2_Pi
        if t <= _Pi and v <= _Pi:
            return True, t, u, v
    return False, None, None, None


def _CCSCC(x, y, phi, sols):
    flag, t, u, v = _LpRmSmLmRp(x, y, phi)
    if flag:  # LpRmSmLmRp
        sols.append(
            {
                "type": ["Lp", "Rm", "Sm", "Lm", "Rp"],
                "lens": [t, _Pi_2, u, _Pi_2, v],
                "cost": t + u + v + _Pi,
            }
        )
    flag, t, u, v = _LpRmSmLmRp(-x, y, -phi)  # timeflip
    if flag:  # LmRpSpLpRm
        sols.append(
            {
                "type": ["Lm", "Rp", "Sp", "Lp", "Rm"],
                "lens": [t, _Pi_2, u, _Pi_2, v],
                "cost": t + u + v + _Pi,
            }
        )
    flag, t, u, v = _LpRmSmLmRp(x, -y, -phi)  # reflect
    if flag:  # RpLmSmRmLp
        sols.append(
            {
                "type": ["Rp", "Lm", "Sm", "Rm", "Lp"],
                "lens": [t, _Pi_2, u, _Pi_2, v],
                "cost": t + u + v + _Pi,
            }
        )
    flag, t, u, v = _LpRmSmLmRp(-x, -y, phi)  # timeflip + reflect
    if flag:  # RmLpSpRpLm
        sols.append(
            {
                "type": ["Rm", "Lp", "Sp", "Rp", "Lm"],
                "lens": [t, _Pi_2, u, _Pi_2, v],
                "cost": t + u + v + _Pi,
            }
        )


@timer(enabled=CAR_PATH_MODULE_TIMER_ENABLED)
def rs_all_paths(x, y, phi, x_, y_, phi_, kappa):
    loc = _corrd_trans(x, y, phi, x_, y_, phi_, kappa)
    sols = []
    _CSC(*loc, sols)
    _CCC(*loc, sols)
    _CCSC(*loc, sols)
    _CSCC(*loc, sols)
    _CCCC(*loc, sols)
    _CCSCC(*loc, sols)
    return sols


@timer(enabled=CAR_PATH_MODULE_TIMER_ENABLED)
def rs_shortest_path(x, y, phi, x_, y_, phi_, kappa):
    """
    params:
        x: x-coordinate of the current position
        y: y-coordinate of the current position
        phi: current heading angle
        x_: x-coordinate of the target position
        y_: y-coordinate of the target position
        phi_: target heading angle
        kappa: curvature of reeds-shepp path
        step: step size of the path interpolation
    returns:
        a dictionary containing the shortest reeds-shepp path
            and the corresponding x, y, and phi coordinates
    """
    return min(
        rs_all_paths(x, y, phi, x_, y_, phi_, kappa),
        key=lambda sol: sol["cost"],
    )


@timer(enabled=CAR_PATH_MODULE_TIMER_ENABLED)
def rs_shortest_path_length(x, y, phi, x_, y_, phi_, kappa):
    """
    params:
        x: x-coordinate of the current position
        y: y-coordinate of the current position
        phi: current heading angle
        x_: x-coordinate of the target position
        y_: y-coordinate of the target position
        phi_: target heading angle
        kappa: curvature of reeds-shepp path
    returns:
        cost: the cost of the path from the current position to the target position
    """
    return rs_shortest_path(x, y, phi, x_, y_, phi_, kappa)["cost"]


def path_plot(x, y, phi, x_, y_, phi_, kappa, step=0.1, ax=None, path_type="dubins"):
    if ax is None:
        ax = plt.gca()
    if path_type == "dubins":
        sols = dubins_all_paths(x, y, phi, x_, y_, phi_, kappa)
    elif path_type == "reeds-shepp":
        sols = rs_all_paths(x, y, phi, x_, y_, phi_, kappa)
    else:
        raise ValueError("Invalid path_type")
    for sol in sols:
        path_x, path_y, path_phi = path_interp(
            sol["type"], sol["lens"], step, kappa, x, y, phi
        )
        ax.plot(path_x, path_y, label=sol["type"])
    opt_sol = min(sols, key=lambda sol: sol["cost"])
    opt_path_x, opt_path_y, opt_path_phi = path_interp(
        opt_sol["type"], opt_sol["lens"], step, kappa, x, y, phi
    )
    ax.scatter(
        opt_path_x,
        opt_path_y,
        label=f"Shortest Path {opt_sol['type']})",
        marker="x",
        color="green",
    )
    ax.arrow(
        x,
        y,
        np.cos(phi),
        np.sin(phi),
        head_width=0.1,
        head_length=0.1,
        color="red",
        label="start",
    )
    ax.arrow(
        x_,
        y_,
        np.cos(phi_),
        np.sin(phi_),
        head_width=0.1,
        head_length=0.1,
        color="green",
        label="goal",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(path_type)
    ax.legend()
    ax.axis("equal")


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Example usage
    x = np.random.uniform(-5, 5, 2)
    y = np.random.uniform(-5, 5, 2)
    phi = np.random.uniform(0, 2 * np.pi, 2)
    kappa = 1 / 2.2
    # dubins_all_paths(x[0], y[0], phi[0], x[1], y[1], phi[1], kappa)
    # dubins_shortest_path(x[0], y[0], phi[0], x[1], y[1], phi[1], kappa)
    # dubins_shortest_path_length(x[0], y[0], phi[0], x[1], y[1], phi[1], kappa)
    # rs_all_paths(x[0], y[0], phi[0], x[1], y[1], phi[1], kappa)
    # rs_shortest_path(x[0], y[0], phi[0], x[1], y[1], phi[1], kappa)
    # rs_shortest_path_length(x[0], y[0], phi[0], x[1], y[1], phi[1], kappa)
    path_plot(x[0], y[0], phi[0], x[1], y[1], phi[1], kappa, path_type="dubins")
    plt.show()
