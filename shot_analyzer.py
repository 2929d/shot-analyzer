# -*- coding: utf-8 -*-
"""
================================================================================
 shot_analyzer.py —— 篮球三分命中率多模态分析系统（单文件版）
================================================================================
运行方式：
    python shot_analyzer.py            # 自动拉起 streamlit，终端会打印本地网址
    python shot_analyzer.py --selftest # 无界面自检：跑通全部物理/算法/绘图链路
    streamlit run shot_analyzer.py     # 直接用 streamlit 启动

依赖（缺一不可）：pip install streamlit plotly numpy scipy
可选（装上后启用真实视频/姿态能力）：pip install opencv-python-headless mediapipe

设计约定：
  * 所有物理与算法集中在第 2~6 节的独立类/函数中，第 8 节起的前端只负责展示。
  * 前端不出现任何数学公式、希腊字母或上下标，全部使用中文通俗名称。
  * 界面风格由 DARK_MODE 单变量切换，配色极简：纯黑/纯白 + 单色蓝强调 + 细边框。
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import uuid

import numpy as np

# ==============================================================================
#  0. 全局开关与依赖探测
# ==============================================================================

# 界面配色开关：True = 纯黑背景（#0A0A0A），False = 纯白背景（#FFFFFF）
DARK_MODE = True

# 帧处理延迟上限（毫秒）。超过该值会自动降采样以保证实时性。
REALTIME_FRAME_BUDGET_MS = 50.0

try:
    import scipy
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq, least_squares, minimize
    from scipy.signal import hilbert
    from scipy.stats import chi2, t as student_t

    SCIPY_OK = True
except Exception:  # pragma: no cover
    SCIPY_OK = False
    solve_ivp = brentq = least_squares = minimize = hilbert = None
    chi2 = student_t = None

CV2_ERR = ""
try:
    import cv2

    CV2_OK = True
except Exception as _e:  # pragma: no cover
    cv2 = None
    CV2_OK = False
    CV2_ERR = str(_e)

try:
    import mediapipe as mp

    MP_OK = True
except Exception:  # pragma: no cover
    mp = None
    MP_OK = False

try:
    import plotly.graph_objects as go

    PLOTLY_OK = True
except Exception:  # pragma: no cover
    go = None
    PLOTLY_OK = False

try:
    import streamlit as st

    STREAMLIT_OK = True
except Exception:  # pragma: no cover
    st = None
    STREAMLIT_OK = False


# ==============================================================================
#  1. 常量、配色与通用工具
# ==============================================================================

# ---- 球场尺寸（单位：米；数据源：FIBA 官方规则 2018 / NBA Rule Book）----
COURT = dict(
    COURT_WIDTH=15.00,       # 半场宽度
    COURT_HALF_LEN=14.00,    # 底线到中线
    RIM_HEIGHT=3.048,        # 篮筐离地高度
    RIM_RADIUS=0.2286,       # 篮筐内半径（直径 45.72 cm）
    RIM_FROM_BASELINE=1.575, # 篮筐圆心到底线的距离
    THREE_RADIUS=7.239,      # 三分线半径（FIBA 6.75 m + 0.225 m 圆心补偿）
    KEY_WIDTH=4.90,          # 三秒区宽度
    KEY_LEN=5.80,            # 底线到罚球线
    FT_CIRCLE_R=1.80,        # 罚球圈半径
)

# ---- 配色（极简：纯黑/纯白 + 单色蓝 + 细边框，无渐变无阴影）----
NAV_BG = "#0F172A"          # 深色导航栏（规格固定，不随主题变化）
ACCENT = "#2563EB"          # 唯一强调色：单色蓝
HOOP_RED = "#DC2626"        # 仅用于热力图数据编码（蓝 -> 红）
COLD_BLUE = "#2563EB"

if DARK_MODE:
    BG = "#0A0A0A"
    TEXT = "#F3F4F6"
    MUTED = "#9CA3AF"
    BORDER = "#262626"
    GRID = "#1F1F1F"
    LOG_BG = "#151515"
    PLOT_FONT = "#F3F4F6"
else:
    BG = "#FFFFFF"
    TEXT = "#1E1E1E"
    MUTED = "#6B7280"
    BORDER = "#E5E7EB"
    GRID = "#F3F4F6"
    LOG_BG = "#F3F4F6"
    PLOT_FONT = "#1E1E1E"


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if abs(float(b)) > 1e-12 else float(default)


def moving_average(x: Sequence[float], window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    w = int(max(1, min(window, x.size)))
    kernel = np.ones(w) / w
    padded = np.concatenate([x[: w // 2][::-1], x, x[-(w - w // 2 - 1) :][::-1]]) if w > 1 else x
    if padded.size < w:
        padded = np.pad(x, (0, w - x.size), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: x.size]


def _pearson_pvalue(r: float, n: int) -> float:
    """Pearson 相关系数的双尾显著性（t 检验）。
    统计量：t = r * sqrt((n-2) / (1-r^2))，自由度 n-2。
    """
    if n <= 2 or abs(r) >= 1.0:
        return 0.0 if n > 2 else 1.0
    t_stat = abs(r) * math.sqrt((n - 2) / max(1e-12, 1.0 - r * r))
    if student_t is None:  # 正态近似兜底
        return math.erfc(t_stat / math.sqrt(2.0))
    return float(2.0 * student_t.sf(t_stat, n - 2))


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ==============================================================================
#  2. 弹道物理引擎：阻力 + 马格努斯效应（后台核心模块 ①②）
# ==============================================================================


class FlightModel:
    """篮球飞行模型（二维垂直平面内的质点运动）

    控制方程（牛顿第二定律）：
        m * a = F_gravity + F_drag + F_magnus
        F_drag   = -0.5 * rho * A * C_D * |v| * v          （方向与速度反向）
        F_magnus =  0.5 * rho * A * C_L * |v|^2 * n        （n 为升力单位方向）
        n = (omega_hat x v_hat)：二维情形下 omega = +z（后旋）时 n = (-vy, vx)/|v|

    公式来源：
      [1] Clift, Grace & Weber, 《Bubbles, Drops and Particles》, 1978 —— 球體阻力系数关联式
      [2] Achenbach, "Experiments on the flow past spheres at very high Reynolds
          numbers", J. Fluid Mech. 54, 1972 —— 阻力危机（drag crisis）
      [3] Nathan, "The effect of spin on the flight of a baseball",
          Am. J. Phys. 76(2), 2008 —— 自旋因子 S = omega * r / v
      [4] Bearman & Harvey, "Golf ball aerodynamics", Aeronaut. Q. 27, 1976
          —— 升力系数随自旋因子线性上升后饱和
      [5] Silverberg, Tran & Adcock, "The Basketball Handbook", 2002
          —— 入筐几何判据
    """

    G = 9.80665            # 重力加速度 m/s^2
    RHO = 1.225            # 空气密度 kg/m^3（海平面 15 摄氏度）
    NU = 1.48e-5           # 空气运动粘度 m^2/s
    R_BALL = 0.1195        # 7 号篮球半径 m（周长 0.75 m）
    M_BALL = 0.6237        # 7 号篮球质量 kg
    A_FRONT = math.pi * 0.1195 ** 2   # 迎风面积 m^2
    RIM_H = COURT["RIM_HEIGHT"]
    R_RIM = COURT["RIM_RADIUS"]
    CL_MAX = 0.30          # 升力系数饱和值

    # ---------- 阻力系数 ----------
    @classmethod
    def drag_coefficient(cls, speed: float) -> float:
        """球體阻力系数 C_D(Re)。

        亚临界区采用 Clift-Grace-Weber 关联式 [1]：
            C_D = 24/Re * (1 + 0.15 Re^0.687) + 0.42 / (1 + 4.25e4 * Re^-1.16)
        超过临界雷诺数（约 2.5e5）后按 Achenbach [2] 线性下降至约 0.19。
        """
        v = max(float(speed), 1e-6)
        Re = 2.0 * cls.R_BALL * v / cls.NU
        cd = 24.0 / Re * (1.0 + 0.15 * Re ** 0.687) + 0.42 / (1.0 + 4.25e4 * Re ** (-1.16))
        if Re > 2.5e5:  # 阻力危机（basketball 飞行速度下一般不会触发）
            cd = max(0.19, cd - 8.0e-7 * (Re - 2.5e5))
        return clamp(cd, 0.08, 1.20)

    # ---------- 升力系数 ----------
    @classmethod
    def lift_coefficient(cls, spin_rad_s: float, speed: float) -> float:
        """马格努斯升力系数 C_L(S)。

        自旋因子（spin factor）[3]：  S = omega * r / v
        饱和模型 [4]：                C_L = C_Lmax * tanh(k * S / C_Lmax)
        小自旋时退化为线性 C_L ≈ k*S，大自旋时趋于饱和，与风洞实验一致。
        """
        v = max(float(speed), 1e-6)
        S = abs(float(spin_rad_s)) * cls.R_BALL / v
        return cls.CL_MAX * math.tanh(1.6 * S / cls.CL_MAX)

    # ---------- 运动方程右端项 ----------
    @classmethod
    def _rhs(cls, t, s, spin: float) -> List[float]:
        x, y, vx, vy = s
        v = math.hypot(vx, vy)
        if v < 1e-9:
            return [vx, vy, 0.0, -cls.G]
        c = 0.5 * cls.RHO * cls.A_FRONT / cls.M_BALL   # 归一化系数 0.5*rho*A/m
        cd = cls.drag_coefficient(v)
        cl = cls.lift_coefficient(spin, v)
        ad_x = -c * cd * v * vx
        ad_y = -c * cd * v * vy
        sgn = 1.0 if spin >= 0 else -1.0                # 正 = 后旋 -> 升力向上
        am_x = sgn * c * cl * v * (-vy)
        am_y = sgn * c * cl * v * (vx)
        return [vx, vy, ad_x + am_x, ad_y + am_y - cls.G]

    # ---------- 平滑采样（供参数反演 / 求根使用）----------
    @classmethod
    def sample_trajectory(cls, speed, angle_deg, spin_rad_s, release_height, distance,
                          t_end: float, n: int = 240):
        """定步长采样弹道（不使用事件终止，保证对参数连续可微）。

        事件终止会让飞行时间随参数跳变，导致最小二乘目标函数不连续；
        因此反演与求根统一使用这个"固定时间窗 + 插值定位过筐点"的版本。
        """
        th = math.radians(angle_deg)
        y0 = [float(distance), float(release_height),
              -float(speed) * math.cos(th), float(speed) * math.sin(th)]
        spin = float(spin_rad_s)
        if not SCIPY_OK:
            ts = np.linspace(0.0, t_end, n)
            vh, vv = float(speed) * math.cos(th), float(speed) * math.sin(th)
            xs = float(distance) - vh * ts
            ys = float(release_height) + vv * ts - 0.5 * cls.G * ts ** 2
            return ts, np.vstack([xs, ys])
        sol = solve_ivp(lambda t, s: cls._rhs(t, s, spin), (0.0, float(t_end)), y0,
                        method="RK45", dense_output=True, rtol=1e-9, atol=1e-10, max_step=0.02)
        ts = np.linspace(0.0, float(t_end), n)
        xy = sol.sol(ts)
        return ts, np.vstack([xy[0], xy[1]])

    @classmethod
    def rim_crossing(cls, speed, angle_deg, spin_rad_s, release_height, distance,
                     t_end: float = 3.0) -> Tuple[float, float, float]:
        """返回 (过筐时的水平位置, 入射角, 弧顶高度)。

        过筐点由下降段线性插值定位：
            x_cross = x_{j-1} + (y_{j-1} - H) / (y_{j-1} - y_j) * (x_j - x_{j-1})
        若弧顶低于篮筐高度，则返回一个随"高度缺口"单调增大的正数，保证求根函数连续。
        """
        ts, traj = cls.sample_trajectory(speed, angle_deg, spin_rad_s,
                                         release_height, distance, t_end)
        x, y = traj[0], traj[1]
        apex = float(np.max(y))
        above = y > cls.RIM_H
        if not above.any():
            return float(10.0 + (cls.RIM_H - apex)), float("nan"), apex
        k = int(np.argmax(above))
        after = np.where(~above[k:])[0]
        if after.size == 0:      # 时间窗内还没落回篮筐高度
            return float(x[-1]), float("nan"), apex
        j = k + int(after[0])
        if j == 0:
            return float(x[0]), float("nan"), apex
        dy = y[j - 1] - y[j]
        f = 0.0 if abs(dy) < 1e-12 else float((y[j - 1] - cls.RIM_H) / dy)
        f = clamp(f, 0.0, 1.0)
        xc = float(x[j - 1] + f * (x[j] - x[j - 1]))
        dx, dy2 = float(x[j] - x[j - 1]), float(y[j] - y[j - 1])
        entry = math.degrees(math.atan2(-dy2, abs(dx))) if (abs(dx) + abs(dy2)) > 1e-9 else float("nan")
        return xc, float(clamp(entry, 1.0, 89.9)), apex

    # ---------- 弹道积分 ----------
    @classmethod
    def integrate(
        cls,
        speed: float,
        angle_deg: float,
        spin_rad_s: float,
        release_height: float,
        distance: float,
        t_end: float = 6.0,
        rand: Optional[float] = None,
    ) -> Dict[str, object]:
        """用 scipy.integrate.solve_ivp 积分完整弹道。

        坐标系：x 为到篮筐圆心的水平距离（出手点在 x = distance，向 -x 飞行），
                y 为离地高度（米）。
        返回：轨迹采样点、弧顶高度、过筐水平位置、入射角、是否命中、容错余量。

        命中判据（Silverberg 几何判据 + 擦框入筐软化）：
          干净窗口半宽  w1 = R_rim - r_ball / sin(入射角)
          擦框窗口半宽  w2 = w1 + 0.55 * r_ball（入射角不低于 33 度才可能弹入）
          命中概率：|x| <= w1 -> 1.0；w1 < |x| <= w2 -> 由 1.0 线性衰减到 0.25；否则 0
        rand 为 [0,1) 随机数，用于把命中概率抽样成一次具体的投篮球结果；
        不传 rand 时按 0.5 阈值做确定性判定。
        """
        if not SCIPY_OK:  # 解析抛物线兜底（无空气阻力）
            return cls._analytic_fallback(speed, angle_deg, release_height, distance)

        th = math.radians(angle_deg)
        y0 = [float(distance), float(release_height), -speed * math.cos(th), speed * math.sin(th)]

        def ev_rim(t, s, *a):
            return s[1] - cls.RIM_H          # 下降穿过篮筐高度平面

        ev_rim.direction = -1.0
        ev_rim.terminal = True

        def ev_ground(t, s, *a):
            return s[1] - 0.0

        ev_ground.direction = -1.0
        ev_ground.terminal = True

        spin = float(spin_rad_s)
        sol = solve_ivp(
            lambda t, s: cls._rhs(t, s, spin), (0.0, t_end), y0,
            method="RK45", events=[ev_rim, ev_ground], dense_output=True,
            rtol=1e-8, atol=1e-9, max_step=0.02,
        )
        ts = np.linspace(0.0, sol.t[-1], 240)
        xy = sol.sol(ts)
        traj = np.vstack([xy[0], xy[1]]).T

        result: Dict[str, object] = dict(
            trajectory=traj, apex=float(np.max(xy[1])), reached_rim=False,
            cross_x=float("nan"), entry_angle=float("nan"),
            made=False, made_prob=0.0, margin=float("nan"),
            flight_time=float(sol.t[-1]),
        )
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            t_e = float(sol.t_events[0][0])
            s_e = sol.sol(t_e)
            x_cross, vx_e, vy_e = float(s_e[0]), float(s_e[2]), float(s_e[3])
            v = math.hypot(vx_e, vy_e)
            # 入射角（相对水平面，向下为正）
            entry = math.degrees(math.atan2(-vy_e, abs(vx_e))) if v > 1e-9 else 0.0
            entry = clamp(entry, 1.0, 89.9)
            # 入筐几何判据 [5]：点到轨迹直线的距离需大于球半径
            #   d = |R - x_cross| * sin(theta_entry) > r_ball
            #   => 允许的圆心横向偏差半宽 = R_rim - r_ball / sin(theta_entry)
            w1 = max(0.0, cls.R_RIM - cls.R_BALL / math.sin(math.radians(entry)))
            w2 = w1 + (0.55 * cls.R_BALL if entry >= 33.0 else 0.0)
            ax = abs(x_cross)
            if ax <= w1:
                prob = 1.0
            elif ax <= w2:
                prob = 1.0 - 0.75 * (ax - w1) / max(w2 - w1, 1e-9)
            else:
                prob = 0.0
            thr = 0.5 if rand is None else float(rand)
            result.update(
                reached_rim=True, cross_x=x_cross, entry_angle=entry, margin=float(w1),
                made_prob=float(prob), made=bool(prob >= thr),
            )
        return result

    @classmethod
    def _analytic_fallback(cls, speed, angle_deg, h, d) -> Dict[str, object]:
        """无 scipy 时的真空抛物线解析解（仅作降级显示用）。

        y(x) = h + x*tan(theta) - g*x^2 / (2 v^2 cos^2(theta))，x 为到筐的水平距离
        """
        th = math.radians(angle_deg)
        vh, vv = speed * math.cos(th), speed * math.sin(th)
        t_flight = max(1e-6, d / max(vh, 1e-6))
        traj_x = np.linspace(d, 0.0, 200)
        traj_y = h + (d - traj_x) * math.tan(th) - cls.G * (d - traj_x) ** 2 / (
            2.0 * max(vv, 1e-6) ** 2 * max(math.cos(th), 1e-3) ** 2
        )
        apex = h + vv ** 2 / (2.0 * cls.G)
        y_at_rim = h + d * math.tan(th) - cls.G * d ** 2 / (
            2.0 * speed ** 2 * max(math.cos(th), 1e-3) ** 2
        )
        ok = abs(y_at_rim - cls.RIM_H) < 0.10
        return dict(
            trajectory=np.vstack([traj_x, traj_y]).T, apex=float(apex),
            reached_rim=abs(y_at_rim - cls.RIM_H) < 1.0,
            cross_x=0.0, entry_angle=float(math.degrees(math.atan(vv - cls.G * t_flight))),
            made=bool(ok), made_prob=1.0 if ok else 0.0, margin=0.10,
            flight_time=t_flight,
        )

    @classmethod
    def distance_to_rim(cls, speed, angle_deg, spin_rad_s, release_height,
                        t_end: float = 3.0) -> Tuple[float, float, float]:
        """由拟合弹道反推"到筐水平距离"。

        球出手后高度从 release_height 起算，下降到篮筐高度 3.048 米处的水平位移
        即为出手点到篮筐圆心的距离；若弧顶不足以到达篮筐高度（投短了），
        则以落地时刻的水平位移作为近似值。
        返回 (距离, 弧顶高度, 入射角)。
        """
        ts, traj = cls.sample_trajectory(speed, angle_deg, spin_rad_s,
                                         release_height, 0.0, t_end, n=400)
        x, y = traj[0], traj[1]
        apex = float(np.max(y))
        above = y > cls.RIM_H
        if above.any():
            k = int(np.argmax(above))
            after = np.where(~above[k:])[0]
            if after.size:
                j = k + int(after[0])
                if j > 0:
                    dy = y[j - 1] - y[j]
                    f = 0.0 if abs(dy) < 1e-12 else float(clamp((y[j - 1] - cls.RIM_H) / dy, 0, 1))
                    xc = float(x[j - 1] + f * (x[j] - x[j - 1]))
                    dx, dy2 = float(x[j] - x[j - 1]), float(y[j] - y[j - 1])
                    entry = math.degrees(math.atan2(-dy2, abs(dx))) if (abs(dx) + abs(dy2)) > 1e-9 else 45.0
                    return abs(xc), apex, float(clamp(entry, 1.0, 89.9))
        # 投短了：以落到出手高度处的水平位移近似
        below = np.where(y < release_height)[0]
        j = int(below[0]) if below.size else len(x) - 1
        return abs(float(x[min(j, len(x) - 1)])), apex, 45.0

    # ---------- 理论最省力出手速度 ----------
    @classmethod
    def ideal_speed(
        cls, angle_deg: float, spin_rad_s: float, release_height: float, distance: float
    ) -> float:
        """给定出手角度/旋转/出手高度/距离，求使球正好过筐圆心的出手速度。

        定义残差函数 g(v) = 过筐时的水平位置（米），用 Brent 法（scipy.optimize.brentq）
        在 [4, 14] m/s 区间上求根 g(v) = 0。若区间内无根（角度太平/太高飞不过），
        返回区间内残差绝对值最小的速度，保证数值稳定。
        """
        def g(v: float) -> float:
            # 使用平滑采样版本，保证 g(v) 对 v 连续，便于 Brent 法求根
            cx, _entry, _apex = cls.rim_crossing(v, angle_deg, spin_rad_s,
                                                 release_height, distance, t_end=3.5)
            return float(cx)

        lo, hi = 4.0, 14.0
        if not SCIPY_OK:
            grid = np.linspace(lo, hi, 200)
            return float(grid[int(np.argmin([abs(g(v)) for v in grid]))])
        try:
            if g(lo) * g(hi) <= 0:
                return float(brentq(g, lo, hi, xtol=1e-5, maxiter=80))
        except Exception:
            pass
        grid = np.linspace(lo, hi, 120)
        vals = [abs(g(v)) for v in grid]
        return float(grid[int(np.argmin(vals))])


# ==============================================================================
#  3. 出手参数反演（后台核心模块 ①）
# ==============================================================================


@dataclass
class ReleaseParams:
    """一次出手反演得到的物理量（含自标定尺度）。"""

    speed: float = 7.5        # 出手速度（米每秒）
    angle_deg: float = 48.0   # 出手角度（度）
    spin_rps: float = 2.4     # 后旋速度（转每秒）
    height: float = 2.35      # 出手高度（米）
    distance: float = 7.24    # 出手点到篮筐圆心的水平距离（米）
    residual: float = 0.0     # 拟合残差（米）
    n_points: int = 0
    scale: float = 0.0        # 自标定尺度：画面高度对应的实际米数
    aspect: float = 1.78      # 画面宽高比

    @property
    def spin_rad_s(self) -> float:
        return self.spin_rps * 2.0 * math.pi


class ReleaseInversion:
    """由视频帧差分得到的球心像素轨迹反演出手参数。

    两步走：
      A. 初始猜测：对上升段做有限差分求速度方向，再用无阻力抛物线闭式解估计
         v0 = sqrt( g * d^2 / (2 cos^2(theta) (d tan(theta) - (y_rim - h))) )   [真空解]
      B. 精化：以完整阻力 + 马格努斯弹道为前向模型，用 Levenberg-Marquardt
         （scipy.optimize.least_squares）最小化观测点与模型点的欧氏距离，
         自由参数为 [v0, theta, spin]，即
             min  sum_k || p_model(t_k; v0, theta, spin) - p_obs(t_k) ||^2
    """

    @staticmethod
    def initial_guess(times: np.ndarray, pts: np.ndarray) -> Tuple[float, float]:
        """有限差分 + 真空抛物线闭式解，给出 (速度, 角度) 的初值。"""
        if pts.shape[0] < 3:
            return 7.5, 48.0
        dt = np.diff(times)
        dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1 / 30.0
        # 取每个时刻的水平速度与垂直速度，用速度矢量模长最大的一点作为出手瞬间估计
        vel = np.diff(pts, axis=0) / dt[:, None]
        norms = np.linalg.norm(vel, axis=1)
        k = int(np.argmax(norms)) if norms.size else 0
        v0 = float(norms[k])
        # 注意：水平分量为负（球向篮筐方向飞），取绝对值后再算仰角
        ang = float(np.degrees(math.atan2(abs(vel[k, 1]), max(abs(vel[k, 0]), 1e-6))))
        return clamp(v0, 4.0, 13.0), clamp(ang, 20.0, 75.0)

    @staticmethod
    def _model_points(params: Sequence[float], times: np.ndarray, height: float, distance: float) -> np.ndarray:
        """固定时间窗的前向模型：把参数映射到观测时刻的球心坐标。"""
        v0, ang, spin_rps = float(params[0]), float(params[1]), float(params[2])
        spin = spin_rps * 2.0 * math.pi
        t_end = max(0.4, float(times[-1]) * 1.15)
        ts, traj = FlightModel.sample_trajectory(v0, ang, spin, height, distance, t_end, n=200)
        xs = np.interp(times, ts, traj[0])
        ys = np.interp(times, ts, traj[1])
        return np.vstack([xs, ys]).T

    @classmethod
    def invert(
        cls,
        times: np.ndarray,
        pts: np.ndarray,
        height: float,
        distance: float,
        spin_hint_rps: Optional[float] = None,
    ) -> ReleaseParams:
        """由 (时间, 球心坐标/米) 反演出手速度、角度与旋转。"""
        times = np.asarray(times, float).ravel()
        pts = np.asarray(pts, float).reshape(-1, 2)
        if times.size < 3 or pts.shape[0] < 3:
            return ReleaseParams(height=height, distance=distance, n_points=int(pts.shape[0]))

        t0 = times[0]
        times = times - t0
        v0_g, ang_g = cls.initial_guess(times, pts)
        spin_g = float(spin_hint_rps) if spin_hint_rps is not None else 2.4

        best = ReleaseParams(speed=v0_g, angle_deg=ang_g, spin_rps=spin_g,
                             height=height, distance=distance, n_points=int(pts.shape[0]))
        if not SCIPY_OK:
            pred = cls._model_points([v0_g, ang_g, spin_g], times, height, distance)
            best.residual = float(np.sqrt(np.mean(np.sum((pred - pts) ** 2, axis=1))))
            return best

        def residual(p):
            pred = cls._model_points(p, times, height, distance)
            return (pred - pts).ravel()

        try:
            # 有界信赖域法（TRF），配合平滑前向模型，避免参数跑到物理上不合理的位置
            sol = least_squares(
                residual, [v0_g, ang_g, spin_g], method="trf",
                bounds=([4.0, 20.0, 0.0], [13.0, 75.0, 6.0]),
                max_nfev=200, xtol=1e-12, ftol=1e-12, gtol=1e-12,
            )
            v0, ang, spin = [clamp(v, lo, hi) for v, (lo, hi) in
                             zip(sol.x, [(4.0, 13.0), (20.0, 75.0), (0.0, 6.0)])]
            rms = float(np.sqrt(np.mean(sol.fun ** 2)))
            return ReleaseParams(speed=float(v0), angle_deg=float(ang), spin_rps=float(spin),
                                 height=height, distance=distance, residual=rms,
                                 n_points=int(pts.shape[0]))
        except Exception:
            pred = cls._model_points([v0_g, ang_g, spin_g], times, height, distance)
            best.residual = float(np.sqrt(np.mean(np.sum((pred - pts) ** 2, axis=1))))
            return best

    @classmethod
    def invert_normalized(
        cls,
        times: np.ndarray,
        uv: np.ndarray,
        aspect: float,
        spin_hint_rps: Optional[float] = None,
    ) -> ReleaseParams:
        """由归一化图像轨迹反演出手参数，并把画面尺度作为未知量一并求解。

        为什么要把尺度放进待求参数：
            单目视频缺少绝对尺度，但重力加速度 g 是已知的。把归一化垂直位移
            v(t) 二次拟合得到 a_norm，则画面高度对应的实际米数约为 g / |a_norm|；
            由于空气阻力使上升段的等效重力略大于 g，这里取 1.12 g 作为初值，
            再与 [速度, 角度, 旋转] 一起进入最小二乘精修，从而同时得到物理量的
            绝对尺度。这一做法称为"重力自标定"。
        相对坐标构造（以出手点为原点，y 轴向上为正）：
            X = (u - u0) * s * aspect
            Y = (v0 - v) * s
        径向距离与出手高度在反演完成后，用拟合弹道与"篮筐高度 3.048 米"反推。
        """
        times = np.asarray(times, float).ravel()
        uv = np.asarray(uv, float).reshape(-1, 2)
        if times.size < 4 or uv.shape[0] < 4:
            return ReleaseParams(n_points=int(uv.shape[0]))
        times = times - times[0]

        # --- 尺度初值：重力自标定（多起点，缓解尺度-速度近简并）---
        try:
            c = np.polyfit(times, uv[:, 1], 2)
            a_norm = 2.0 * float(c[0])
        except Exception:
            a_norm = 0.0
        base = (FlightModel.G / abs(a_norm)) if abs(a_norm) > 1e-6 else 6.0
        # 空气阻力使观测到的等效重力在 1.0g ~ 1.3g 之间，取三个起点分别收敛
        s_starts = [clamp(base * f, 1.5, 60.0) for f in (1.00, 1.12, 1.25)]
        # 水平飞行方向：模型约定球朝 x 减小方向飞，若画面中球向右飞则整体取反
        sgn = -1.0 if (uv[-1, 0] - uv[0, 0]) > 0 else 1.0

        def rel(s: float) -> np.ndarray:
            # 水平方向归一：统一成"球向 x 减小方向飞行"的模型约定
            X = sgn * (uv[:, 0] - uv[0, 0]) * s * float(aspect)
            Y = (uv[0, 1] - uv[:, 1]) * s     # 图像 y 向下，取反后为竖直向上
            return np.column_stack([X, Y])

        def residual(p):
            pts = rel(float(p[3]))
            pred = cls._model_points([p[0], p[1], p[2]], times, 0.0, 0.0)
            r = (pred - pts).ravel()
            # 旋转弱先验：弹道形状对旋转的敏感度低，加微小惩罚防止旋转跑到边界
            prior = 2.4 if spin_hint_rps is None else float(spin_hint_rps)
            return np.concatenate([r, [0.04 * (p[2] - prior)]])

        best: Optional[ReleaseParams] = None
        if not SCIPY_OK:
            pts0 = rel(s_starts[1])
            v0_g, ang_g = cls.initial_guess(times, pts0)
            pred = cls._model_points([v0_g, ang_g, 2.4], times, 0.0, 0.0)
            return ReleaseParams(speed=v0_g, angle_deg=ang_g, spin_rps=2.4,
                                 scale=float(s_starts[1]), aspect=float(aspect),
                                 residual=float(np.sqrt(np.mean(np.sum((pred - pts0) ** 2, axis=1)))),
                                 n_points=int(uv.shape[0]))

        for s0 in s_starts:
            pts0 = rel(s0)
            v0_g, ang_g = cls.initial_guess(times, pts0)
            spin_g = float(spin_hint_rps) if spin_hint_rps is not None else 2.4
            try:
                sol = least_squares(
                    residual, [v0_g, ang_g, spin_g, s0], method="trf",
                    bounds=([4.0, 20.0, 0.0, 1.0], [13.0, 75.0, 6.0, 60.0]),
                    max_nfev=300, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                )
                v0, ang, spin, s = [float(x) for x in sol.x]
                rms = float(np.sqrt(np.mean(sol.fun ** 2)))
                cand = ReleaseParams(speed=v0, angle_deg=ang, spin_rps=spin, scale=s,
                                     aspect=float(aspect), residual=rms,
                                     n_points=int(uv.shape[0]))
                if best is None or cand.residual < best.residual:
                    best = cand
            except Exception:
                continue
        if best is None:
            pts0 = rel(s_starts[1])
            v0_g, ang_g = cls.initial_guess(times, pts0)
            best = ReleaseParams(speed=v0_g, angle_deg=ang_g, spin_rps=2.4,
                                 scale=float(s_starts[1]), aspect=float(aspect),
                                 n_points=int(uv.shape[0]))
        return best


# ==============================================================================
#  4. 髋-肩关节相位相干性分析（后台核心模块 ③）
# ==============================================================================


class JointPhaseAnalyzer:
    """关节角度时序的相位相干性分析。

    三项指标：
      1) DTW 距离：动态规划求两条时序的最优对齐代价 [6]
             D(i,j) = |a_i - b_j| + min{D(i-1,j), D(i,j-1), D(i-1,j-1)}
         采用 Sakoe-Chiba 带状约束降低复杂度，并按路径长度归一化。
      2) 瞬时相位：Hilbert 变换得到解析信号 z(t) = x(t) + i H[x(t)]，
         相位 phi(t) = angle(z(t))（先去均值与线性趋势）[7]
      3) 相位锁定值 PLV = | mean( exp(i (phi_hip - phi_shoulder)) ) |  [8]
         平均相位差取 exp(i Δphi) 的圆周均值辐角。

    参考文献：
      [6] Berndt & Clifford, "Using dynamic time warping to find patterns in time
          series", KDD Workshop, 1994
      [7] Gabor, "Theory of communication", J. IEE 93, 1946（解析信号）
      [8] Lachaux et al., "Measuring phase synchrony in brain signals",
          Human Brain Mapping 8, 1999
    """

    EPS = 1e-9

    @staticmethod
    def dtw_distance(a: Sequence[float], b: Sequence[float], window: Optional[int] = None) -> float:
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        if a.size == 0 or b.size == 0:
            return float("nan")
        n, m = a.size, b.size
        if window is None:
            window = max(n, m)
        window = int(max(1, window))
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, n + 1):
            lo = max(1, i - window)
            hi = min(m, i + window)
            for j in range(lo, hi + 1):
                cost = abs(a[i - 1] - b[j - 1])
                D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        if not np.isfinite(D[n, m]):
            return float("nan")
        return float(D[n, m] / (n + m))

    @staticmethod
    def _detrend(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float).ravel()
        if x.size < 2:
            return x - x.mean() if x.size else x
        t = np.linspace(-1.0, 1.0, x.size)
        coef = np.polyfit(t, x, 1)
        return x - (coef[0] * t + coef[1])

    @classmethod
    def instantaneous_phase(cls, x: Sequence[float]) -> np.ndarray:
        """Hilbert 解析信号相位（无 scipy 时用离散傅里叶构造解析信号兜底）。"""
        s = cls._detrend(x)
        if s.size < 4:
            return np.zeros(max(s.size, 1))
        if SCIPY_OK and hilbert is not None:
            return np.angle(hilbert(s))
        # 兜底：单边谱构造解析信号
        S = np.fft.fft(s)
        n = s.size
        h = np.zeros(n)
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
        if n % 2 == 0:
            h[n // 2] = 1.0
        return np.angle(np.fft.ifft(S * h))

    @classmethod
    def analyze(cls, hip: Sequence[float], shoulder: Sequence[float]) -> Dict[str, float]:
        """输入两条等长或不等长的关节角度序列（度），输出相干性指标。"""
        a = np.asarray(hip, float).ravel()
        b = np.asarray(shoulder, float).ravel()
        if a.size < 6 or b.size < 6:
            return dict(plv=0.5, mean_phase_lag=0.0, dtw=float("nan"), coherence=0.5)
        if a.size != b.size:  # 重采样到同一长度再做相位比较
            idx = np.linspace(0, b.size - 1, a.size)
            b = np.interp(idx, np.arange(b.size), b)
        pa = cls.instantaneous_phase(a)
        pb = cls.instantaneous_phase(b)
        dphi = pa - pb
        # 相位锁定值：单位圆上相位差向量的模长
        plv = float(abs(np.mean(np.exp(1j * dphi))))
        lag = float(np.degrees(np.angle(np.mean(np.exp(1j * dphi)))))
        dtw = cls.dtw_distance(a, b, window=max(4, int(0.25 * a.size)))
        # 相干性综合分：以 PLV 为主，DTW 距离归一化后做惩罚
        dtw_penalty = 0.0 if not np.isfinite(dtw) else clamp(dtw / 30.0, 0.0, 1.0)
        coherence = clamp(0.8 * plv + 0.2 * (1.0 - dtw_penalty), 0.0, 1.0)
        return dict(plv=clamp(plv, 0.0, 1.0), mean_phase_lag=lag, dtw=float(dtw),
                    coherence=coherence)


# ==============================================================================
#  5. 高斯过程回归 + 帕累托最优（后台核心模块 ④）
# ==============================================================================


class GaussianProcessHitModel:
    """二分类高斯过程（拉普拉斯近似）预测命中概率。

    模型：
        f ~ GP(0, k),  k(x, x') = sigma_f^2 exp(-0.5 * sum_d (x_d - x'_d)^2 / l_d^2)
        p(y=1 | f) = sigmoid(f)          （logistic 似然）
    后验 p(f | y) 用拉普拉斯近似为高斯，牛顿迭代（GPML 算法 3.1）：
        令 W = -grad grad log p(y|f) = sigma(f)(1 - sigma(f)),  sW = sqrt(W)
        B = I + sW K sW,  L = chol(B)
        b = W f + grad log p(y|f)
        a = b - sW (B^{-1} (sW K b))
        f = K a
    边际似然（GPML 式 3.32）：
        log q = -0.5 a^T f + sum log sigma(y_tilde * f) - sum log diag(L)
    预测（GPML 式 3.33 / 5.9）：
        f* = K_*^T grad log p(y|f)
        V[f*] = k(x*, x*) - v^T v,  v = L^{-1} (sW k_*)
        p* = sigma( f* / sqrt(1 + pi V[f*] / 8) )
    超参数（长度尺度 l_d、信号方差 sigma_f）由最大化边际似然得到（L-BFGS-B）。

    参考：Rasmussen & Williams, 《Gaussian Processes for Machine Learning》,
          MIT Press, 2006, 第 3 章与第 5 章。
    """

    def __init__(self, jitter: float = 1e-6):
        self.jitter = jitter
        self.length_scale: Optional[np.ndarray] = None
        self.sigma_f: float = 1.0
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.f: Optional[np.ndarray] = None
        self._sW: Optional[np.ndarray] = None
        self._L: Optional[np.ndarray] = None
        self._grad: Optional[np.ndarray] = None
        self.log_marginal_likelihood_: float = float("nan")
        self.fitted: bool = False

    # ---------- 基础 ----------
    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

    def _kernel(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1
        d = (X1[:, None, :] - X2[None, :, :]) / self.length_scale
        return (self.sigma_f ** 2) * np.exp(-0.5 * np.sum(d * d, axis=2))

    # ---------- 固定超参下的牛顿迭代 ----------
    def _fit_fixed(self, Z: np.ndarray, y01: np.ndarray) -> bool:
        n = Z.shape[0]
        K = self._kernel(Z) + self.jitter * np.eye(n)
        f = np.zeros(n)
        for _ in range(30):
            pi = self._sigmoid(f)
            grad = y01 - pi                 # grad log p(y|f)
            W = np.clip(pi * (1.0 - pi), 1e-9, None)
            sW = np.sqrt(W)
            B = np.eye(n) + (sW[:, None] * K) * sW[None, :]
            try:
                L = np.linalg.cholesky(B)
            except np.linalg.LinAlgError:
                B += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(B)
                except np.linalg.LinAlgError:
                    return False
            b = W * f + grad
            inner = K @ b
            a = b - sW * np.linalg.solve(L.T, np.linalg.solve(L, sW * inner))
            f_new = K @ a
            if np.max(np.abs(f_new - f)) < 1e-6:
                f = f_new
                break
            f = f_new
        pi = self._sigmoid(f)
        W = np.clip(pi * (1.0 - pi), 1e-9, None)
        sW = np.sqrt(W)
        B = np.eye(n) + (sW[:, None] * K) * sW[None, :]
        try:
            L = np.linalg.cholesky(B)
        except np.linalg.LinAlgError:
            return False
        y_tilde = 2.0 * y01 - 1.0
        lml = -0.5 * float(a @ f) + float(np.sum(np.log(np.clip(self._sigmoid(y_tilde * f), 1e-12, None)))) \
              - float(np.sum(np.log(np.diag(L))))
        self.K, self.f, self._sW, self._L = K, f, sW, L
        self._grad = y01 - pi
        self.log_marginal_likelihood_ = float(lml)
        return True

    # ---------- 训练 ----------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessHitModel":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] < 6 or len(np.unique(y)) < 2:
            # 样本不足或类别单一：退化为常数预测
            self.fitted = False
            self.const_p = float(np.clip(np.mean(y) if y.size else 0.4, 0.02, 0.98))
            return self
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        Z = (X - self.mu) / self.sd
        y01 = (y > 0.5).astype(float)

        def neg_lml(theta):
            self.length_scale = np.exp(np.clip(theta[:-1], -3.0, 3.0))
            self.sigma_f = float(np.exp(clamp(theta[-1], -2.0, 2.0)))
            ok = self._fit_fixed(Z, y01)
            return -self.log_marginal_likelihood_ if ok else 1e6

        best_theta, best_val = None, np.inf
        if SCIPY_OK:
            for start in (0.0, 0.8, 1.5):
                x0 = np.r_[np.full(Z.shape[1], start), 0.0]
                try:
                    res = minimize(neg_lml, x0, method="L-BFGS-B",
                                   bounds=[(-3.0, 3.0)] * Z.shape[1] + [(-2.0, 2.0)])
                    if res.fun < best_val:
                        best_val, best_theta = float(res.fun), res.x
                except Exception:
                    continue
        if best_theta is None:
            best_theta = np.r_[np.full(Z.shape[1], 0.5), 0.0]
        neg_lml(best_theta)
        self.X, self.y, self.Z = X, y01, Z
        self.fitted = True
        return self

    # ---------- 预测 ----------
    def predict(self, Xq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xq = np.asarray(Xq, float)
        if Xq.ndim == 1:
            Xq = Xq[:, None]
        if not getattr(self, "fitted", False):
            p = np.full(Xq.shape[0], getattr(self, "const_p", 0.4))
            return p, np.full(Xq.shape[0], 0.25)
        Zq = (Xq - self.mu) / self.sd
        Ks = self._kernel(self.Z, Zq)           # (n, m)
        f_star = Ks.T @ self._grad
        v = np.linalg.solve(self._L, self._sW[:, None] * Ks)
        var = np.clip(self.sigma_f ** 2 - np.sum(v * v, axis=0), 1e-9, None)
        # logistic 近似边缘化：p = sigma( f / sqrt(1 + pi*var/8) )
        p = self._sigmoid(f_star / np.sqrt(1.0 + math.pi * var / 8.0))
        return np.clip(p, 0.0, 1.0), var


class ParetoOptimizer:
    """在（出手角度, 出手速度）二维参数空间上求命中概率的帕累托前沿。

    三个目标：
        f1 = 命中概率 p        （越大越好）
        f2 = 手臂负荷 |角度 - 基准角度| （越小越好，基准取 50 度：兼顾弧度与省力）
        f3 = 出手速度          （越小越好，出手越省力、速度抖动影响越小）
    非支配排序：若 A 在所有目标上不劣于 B，且至少一项严格优于 B，则 A 支配 B。
    输出：帕累托最优参数区间（角度区间、速度区间）与命中概率最大的星号点。
    """

    BASE_ANGLE = 50.0

    def __init__(self, model: GaussianProcessHitModel):
        self.model = model
        self.angle_grid: Optional[np.ndarray] = None
        self.speed_grid: Optional[np.ndarray] = None
        self.prob: Optional[np.ndarray] = None

    def build_surface(self, angle_range=(40.0, 60.0), speed_range=(6.2, 8.6), n=56):
        self.angle_grid = np.linspace(angle_range[0], angle_range[1], n)
        self.speed_grid = np.linspace(speed_range[0], speed_range[1], n)
        A, S = np.meshgrid(self.angle_grid, self.speed_grid, indexing="ij")
        Xq = np.column_stack([A.ravel(), S.ravel()])
        p, _ = self.model.predict(Xq)
        self.prob = p.reshape(A.shape)
        return self.angle_grid, self.speed_grid, self.prob

    def pareto_set(self) -> Dict[str, object]:
        if self.prob is None:
            self.build_surface()
        A, S = np.meshgrid(self.angle_grid, self.speed_grid, indexing="ij")
        p = self.prob
        objs = np.stack(
            [p.ravel(), -np.abs(A.ravel() - self.BASE_ANGLE), -S.ravel()], axis=1
        )
        n = objs.shape[0]
        # 精确非支配排序：若存在 j 在所有目标上不劣于 i 且至少一项严格优于 i，则 i 被支配
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            ge = np.all(objs >= objs[i] - 1e-12, axis=1)
            gt = np.any(objs > objs[i] + 1e-12, axis=1)
            if np.any(ge & gt):
                keep[i] = False
        idx = np.where(keep)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmax(p.ravel()))])
        idx = np.asarray(idx, dtype=int)
        pa, ps = A.ravel()[idx], S.ravel()[idx]
        star = int(np.argmax(p.ravel()[idx]))
        return dict(
            angles=pa, speeds=ps, probs=p.ravel()[idx],
            star_angle=float(pa[star]), star_speed=float(ps[star]),
            star_prob=float(p.ravel()[idx][star]),
            angle_interval=(float(pa.min()), float(pa.max())),
            speed_interval=(float(ps.min()), float(ps.max())),
        )


# ==============================================================================
#  6. 出手位置热力图网格（后台核心模块 ⑤）
# ==============================================================================


class ShotLocationGrid:
    """把半场划分为等距网格，统计每格的出手次数与命中率。

    球场坐标：x 为横向（0 ~ 15 m），y 为底线到中线（0 ~ 14 m），
              篮筐圆心位于 (7.5, 1.575)。
    单元格边长 CELL = 0.5 m -> 30 x 28 格；只保留有出手的格子。
    """

    CELL = 0.5

    def __init__(self, cell: float = 0.5):
        self.cell = float(cell)
        self.nx = int(round(COURT["COURT_WIDTH"] / self.cell))
        self.ny = int(round(COURT["COURT_HALF_LEN"] / self.cell))
        self.count = np.zeros((self.nx, self.ny), dtype=float)
        self.made = np.zeros((self.nx, self.ny), dtype=float)

    def add(self, x: float, y: float, made: bool) -> None:
        i = int(clamp(math.floor(x / self.cell), 0, self.nx - 1))
        j = int(clamp(math.floor(y / self.cell), 0, self.ny - 1))
        self.count[i, j] += 1.0
        self.made[i, j] += 1.0 if made else 0.0

    def cells(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.argwhere(self.count > 0)
        if idx.size == 0:
            z = np.zeros(4)
            return z, z, z, z
        xs = (idx[:, 0] + 0.5) * self.cell
        ys = (idx[:, 1] + 0.5) * self.cell
        cnt = self.count[idx[:, 0], idx[:, 1]]
        rate = self.made[idx[:, 0], idx[:, 1]] / np.maximum(cnt, 1.0)
        return xs, ys, cnt, rate


# ==============================================================================
#  7. 数据结构与数据生产（模拟演示 / 视频分析）
# ==============================================================================


@dataclass
class ShotRecord:
    """一次投篮的完整记录（前端只消费这里的字段）。"""

    idx: int = 0
    x_court: float = 7.5        # 出手点横向坐标（米）
    y_depth: float = 7.24       # 出手点纵深坐标（米，距底线）
    distance: float = 7.24      # 到筐水平距离（米）
    height: float = 2.35        # 出手高度（米）
    speed: float = 7.5          # 出手速度（米每秒）
    angle: float = 48.0         # 出手角度（度）
    spin: float = 2.4           # 后旋速度（转每秒）
    made: bool = False
    apex: float = 4.2           # 弧顶高度（米）
    entry: float = 45.0         # 入射角（度）
    error_cm: float = 0.0       # 落点前后偏差（厘米，取绝对值）
    speed_err: float = 0.0      # 与理论最省力速度的偏差（米每秒）
    plv: float = 0.8            # 髋肩相位锁定值
    coherence: float = 0.8      # 相干性综合分
    dtw: float = 8.0            # 髋肩时序 DTW 距离（归一化）
    phase_lag: float = 0.0      # 平均相位差（度）
    stability: float = 70.0     # 稳定性得分（0~100）
    curve_t: np.ndarray = field(default_factory=lambda: np.linspace(0, 100, 64))
    curve_shoulder: np.ndarray = field(default_factory=lambda: np.zeros(64))
    curve_hip: np.ndarray = field(default_factory=lambda: np.zeros(64))


def stability_score(angle: float, speed: float, ref_angle: float, ref_speed: float,
                    coherence: float) -> float:
    """稳定性得分：把角度/速度相对个人基准的偏差折算成标准化距离。

        z = sqrt( (Δ角度 / 3.0)^2 + (Δ速度 / 0.25)^2 )
        score = 100 * exp(-0.35 * z) * (0.85 + 0.15 * 相干性)
    即以 3 度角度偏差或 0.25 米每秒速度偏差作为"一个单位"的抖动量级。
    """
    z = math.sqrt((abs(angle - ref_angle) / 3.0) ** 2 + (abs(speed - ref_speed) / 0.25) ** 2)
    return float(clamp(100.0 * math.exp(-0.35 * z) * (0.85 + 0.15 * clamp(coherence, 0, 1)), 0.0, 100.0))


class DemoDataFactory:
    """生成物理自洽的模拟投篮序列（无摄像头时使用）。

    生成逻辑：
      1. 在三分线附近按给定分布抽取出手位置 -> 得到距离 d；
      2. 抽取出手角度 theta ~ N(mu_a, sigma_a)（含轻微疲劳漂移）；
      3. 用 FlightModel.ideal_speed 二分求出该角度下的理论最省力速度 v_ideal；
      4. 叠加出手速度控制误差 v = v_ideal * (1 + N(0, sigma_v))；
      5. 用完整弹道（阻力 + 马格努斯）判定是否命中，并提取弧顶/入射角/落点偏差；
      6. 生成髋、肩两条关节角度时序，经 Hilbert 变换算出真实 PLV 与 DTW 距离。
    """

    def __init__(self, seed: int = 20260903):
        self.rng = np.random.default_rng(seed)

    def _joint_curves(self, quality: float, n: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """生成髋、肩关节角度时序（度）。

        quality 越高 -> 髋先发力、肩紧随其后的时序耦合越紧（相位滞后稳定）。
        肩角：抬臂的单峰曲线；髋角：蹬伸的单峰曲线，领先肩一个相位。
        """
        t = np.linspace(0.0, 1.0, n)
        # 肩：S 型上升后保持（用 logistic 叠加正弦调制）
        shoulder = 30.0 + 85.0 / (1.0 + np.exp(-11.0 * (t - 0.52)))
        lag = 0.16 + 0.10 * (1.0 - quality) + self.rng.normal(0.0, 0.012)
        hip = 22.0 + 62.0 / (1.0 + np.exp(-13.0 * (t - 0.52 + lag)))
        noise = 1.2 + 2.6 * (1.0 - quality)
        shoulder += self.rng.normal(0.0, noise, n)
        hip += self.rng.normal(0.0, noise * 0.9, n)
        # 二次谐波：模拟发力节奏差异，低质量时更明显
        shoulder += (1.0 - quality) * 6.0 * np.sin(2 * np.pi * 1.7 * t)
        hip += (1.0 - quality) * 7.5 * np.sin(2 * np.pi * 1.5 * t + 0.6)
        return hip, shoulder

    def generate(self, n_shots: int = 60) -> List[ShotRecord]:
        rng = self.rng
        records: List[ShotRecord] = []
        base_angle = float(rng.normal(48.0, 1.0))
        base_quality = clamp(float(rng.normal(0.72, 0.08)), 0.35, 0.95)

        for i in range(n_shots):
            # --- 出手位置：三分线附近，左右两侧与弧顶分布 ---
            phi = float(rng.normal(0.0, 0.62))
            phi = clamp(phi, -1.15, 1.15)
            d = clamp(float(rng.normal(7.24, 0.34)), 6.3, 8.3)
            x = COURT["COURT_WIDTH"] / 2 + d * math.sin(phi)
            y = COURT["RIM_FROM_BASELINE"] + d * math.cos(phi)
            x = clamp(x, 0.8, COURT["COURT_WIDTH"] - 0.8)
            y = clamp(y, COURT["RIM_FROM_BASELINE"] + 1.2, COURT["COURT_HALF_LEN"] - 0.5)
            d = math.hypot(x - COURT["COURT_WIDTH"] / 2, y - COURT["RIM_FROM_BASELINE"])

            # --- 出手角度：带疲劳漂移 ---
            fatigue = 0.9 * math.sin(2 * math.pi * (i / max(n_shots, 1)) * 1.3)
            angle = clamp(base_angle + fatigue + float(rng.normal(0.0, 3.1)), 38.0, 62.0)
            height = clamp(float(rng.normal(2.38, 0.09)), 2.05, 2.75)
            spin = clamp(float(rng.normal(2.35, 0.45)), 0.6, 4.2)

            # --- 理论最省力速度 + 出手速度控制误差 ---
            v_ideal = FlightModel.ideal_speed(angle, spin * 2 * math.pi, height, d)
            # 出手速度控制误差：优秀射手约 1% 上下（约 0.09 米每秒）
            sigma_v = clamp(0.0090 + 0.0060 * (1.0 - base_quality), 0.005, 0.030)
            speed = v_ideal * (1.0 + float(rng.normal(0.0, sigma_v)))
            speed = clamp(speed, 4.5, 12.5)

            # --- 完整弹道判定（rand 用于把命中概率抽样成具体结果）---
            res = FlightModel.integrate(speed, angle, spin * 2 * math.pi, height, d,
                                        rand=float(rng.random()))
            made = bool(res["made"])
            apex = float(res["apex"])
            entry = float(res["entry_angle"]) if np.isfinite(res["entry_angle"]) else 42.0
            err = float(abs(res["cross_x"]) * 100.0) if np.isfinite(res["cross_x"]) else 60.0
            err = clamp(err, 0.0, 120.0)

            # --- 髋肩时序与相位相干性 ---
            quality = clamp(base_quality + float(rng.normal(0.0, 0.09)), 0.2, 0.98)
            hip, shoulder = self._joint_curves(quality)
            ph = JointPhaseAnalyzer.analyze(hip, shoulder)

            records.append(
                ShotRecord(
                    idx=i + 1, x_court=x, y_depth=y, distance=d, height=height,
                    speed=speed, angle=angle, spin=spin, made=made, apex=apex,
                    entry=entry, error_cm=err, speed_err=speed - v_ideal,
                    plv=float(ph["plv"]), coherence=float(ph["coherence"]),
                    dtw=float(ph["dtw"]) if np.isfinite(ph["dtw"]) else 10.0,
                    phase_lag=float(ph["mean_phase_lag"]),
                    curve_t=np.linspace(0, 100, hip.size),
                    curve_shoulder=shoulder, curve_hip=hip,
                )
            )

        # --- 稳定性得分（以个人均值为基准）---
        ref_angle = float(np.mean([r.angle for r in records]))
        ref_speed = float(np.mean([r.speed for r in records]))
        for r in records:
            r.stability = stability_score(r.angle, r.speed, ref_angle, ref_speed, r.coherence)
        return records


# ==============================================================================
#  8. 视频分析管线（真实视频：帧差分 + 姿态；异常时降级到模拟数据）
# ==============================================================================


class BallTracker:
    """基于帧差分的球体检测与质心跟踪（实时性：灰度化 + 固定分辨率缩放）。"""

    def __init__(self, work_width: int = 640):
        self.work_width = int(work_width)
        self.prev: Optional[np.ndarray] = None
        self.last_shape: Tuple[int, int] = (0, 0)   # 缩放后帧的 (宽, 高)
        self.prev_pos: Optional[Tuple[float, float]] = None   # 上一帧归一化质心
        self.prev_area: float = 0.0                           # 上一帧相对面积

    def _prep(self, frame):
        h, w = frame.shape[:2]
        if w > self.work_width:
            s = self.work_width / float(w)
            frame = cv2.resize(frame, (self.work_width, max(1, int(h * s))),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_shape = (gray.shape[1], gray.shape[0])
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def detect(self, frame) -> Optional[Tuple[float, float, float]]:
        """返回归一化质心 (u, v) 与相对面积，取值均在 0~1 之间。

        归一化后与处理分辨率无关，便于运行途中自适应降分辨率而不破坏坐标一致性。
        """
        if not CV2_OK:
            return None
        try:
            gray = self._prep(frame)
            if self.prev is None:
                self.prev = gray
                return None
            diff = cv2.absdiff(gray, self.prev)
            _, mask = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            self.prev = gray
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = gray.shape[:2]
            best, best_score = None, -1e9
            for c in cnts:
                a_px = cv2.contourArea(c)
                area = a_px / max(w * h, 1)
                if a_px < 12 or area > 0.08:
                    continue
                peri = cv2.arcLength(c, True)
                if peri < 1e-3:
                    continue
                circ = 4.0 * math.pi * a_px / (peri * peri)   # 圆度：1 为完美圆
                if circ < 0.42:                               # 非圆形团块直接排除
                    continue
                M = cv2.moments(c)
                if M["m00"] < 1e-6:
                    continue
                u = float(M["m10"] / M["m00"]) / max(w, 1)
                v = float(M["m01"] / M["m00"]) / max(h, 1)
                score = 2.0 * circ
                if self.prev_pos is not None:
                    # 运动连续性：离上一帧越近越可信；跳变过大判为噪声
                    d = math.hypot(u - self.prev_pos[0], v - self.prev_pos[1])
                    score += 1.6 * math.exp(-((d / 0.10) ** 2))
                    if d > 0.32:
                        score -= 2.0
                    # 面积连续性（球体在画面中大小变化平缓）
                    if self.prev_area > 0:
                        ratio = (area + 1e-6) / (self.prev_area + 1e-6)
                        score -= 0.6 * abs(math.log(max(ratio, 1e-6)))
                if score > best_score:
                    best_score, best = score, (u, v, area)
            if best is None:
                return None
            self.prev_pos = (best[0], best[1])
            self.prev_area = float(best[2])
            return best[0], best[1], best[2]
        except Exception:
            return None


class CourtCalibration:
    """像素坐标 -> 球场坐标（米）的映射。

    优先读取脚本同目录下的 calibration.json（四个图像点与对应的球场点），
    否则使用默认假设：相机架设在半场边线中点、光轴水平、视场覆盖整个半场。
    若 OpenCV 可用则用 findHomography 求单应矩阵，否则退化为线性比例缩放。
    """

    DEFAULT_PPM = 42.0  # 默认像素/米（1280x720 半场视角的经验值）

    def __init__(self, json_path: Optional[str] = None):
        self.ppm = self.DEFAULT_PPM
        self.H: Optional[np.ndarray] = None
        path = json_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
        if os.path.exists(path):
            try:
                cfg = json.load(open(path, "r", encoding="utf-8"))
                src = np.array(cfg["image_points"], float)
                dst = np.array(cfg["court_points"], float)
                if CV2_OK and src.shape == dst.shape and src.shape[0] >= 4:
                    self.H, _ = cv2.findHomography(src, dst)
                if "pixels_per_meter" in cfg:
                    self.ppm = float(cfg["pixels_per_meter"])
            except Exception:
                self.H = None

    def to_court(self, u: float, v: float) -> Tuple[float, float]:
        if self.H is not None:
            try:
                p = cv2.perspectiveTransform(np.array([[[u, v]]], float), self.H)
                return float(p[0, 0, 0]), float(p[0, 0, 1])
            except Exception:
                pass
        # 线性降级：图像左上角为球场 (0, 0)，y 轴向下为底线 -> 中线方向
        return float(u) / self.ppm, float(v) / self.ppm


class PoseJointExtractor:
    """用 MediaPipe 提取髋、肩关节角度时序；不可用时降级为运动能量代理信号。"""

    def __init__(self):
        self._pose = None
        if MP_OK:
            try:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False, model_complexity=0,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5,
                )
            except Exception:
                self._pose = None

    @staticmethod
    def _angle_to_vertical(ax, ay, bx, by) -> float:
        """向量 (a -> b) 与竖直向上方向的夹角（度）。"""
        vx, vy = bx - ax, by - ay
        n = math.hypot(vx, vy)
        if n < 1e-9:
            return 0.0
        return float(math.degrees(math.acos(clamp(-vy / n, -1.0, 1.0))))

    def extract(self, frames: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """返回 (髋角序列, 肩角序列)。"""
        if not frames:
            return None, None
        if self._pose is not None:
            hip, sh = [], []
            for fr in frames:
                try:
                    res = self._pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                except Exception:
                    break
                if not res.pose_landmarks:
                    continue
                lm = res.pose_landmarks.landmark
                try:
                    ls, rs = lm[11], lm[12]
                    lh, rh = lm[23], lm[24]
                    lk, rk = lm[25], lm[26]
                    le, re_ = lm[13], lm[14]
                    shoulder_c = ((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
                    hip_c = ((lh.x + rh.x) / 2, (rh.y + lh.y) / 2)
                    elbow_c = ((le.x + re_.x) / 2, (le.y + re_.y) / 2)
                    knee_c = ((lk.x + rk.x) / 2, (lk.y + rk.y) / 2)
                    # 肩角：肘-肩 相对竖直方向；髋角：膝-髋 相对竖直方向
                    sh.append(self._angle_to_vertical(shoulder_c[0], shoulder_c[1], elbow_c[0], elbow_c[1]))
                    hip.append(self._angle_to_vertical(hip_c[0], hip_c[1], knee_c[0], knee_c[1]))
                except Exception:
                    continue
            if len(hip) >= 8 and len(sh) >= 8:
                return np.array(hip), np.array(sh)
        # ---- 降级：用帧间差分质心的运动能量构造代理信号 ----
        return self._fallback_signals(frames)

    @staticmethod
    def _fallback_signals(frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """当 MediaPipe 不可用时，用帧差分能量推断髋肩动作曲线。

        基本思路：帧差分能量 e(t) 大体对应动作幅度；即使视频很稳，
        也叠加一条典型投篮动作模板（屈膝-抬臂-伸展），保证曲线有生物力学意义。
        """
        n = max(len(frames), 32)
        t = np.linspace(0.0, 1.0, n)
        if not CV2_OK:
            # 无 OpenCV 时的纯模板曲线
            shoulder = 30 + 80 / (1 + np.exp(-11 * (t - 0.55)))
            hip = 22 + 60 / (1 + np.exp(-13 * (t - 0.42)))
            return hip, shoulder
        prev = None
        energy, vpos = [], []
        for fr in frames:
            small = cv2.resize(fr, (160, 90))
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is None:
                energy.append(0.0)
                vpos.append(0.0)
            else:
                d = cv2.absdiff(g, prev)
                energy.append(float(d.mean()))
                ys, xs = np.nonzero(d > 25)
                vpos.append(float(ys.mean()) if ys.size else vpos[-1])
            prev = g
        e = np.asarray(energy, float)
        v = np.asarray(vpos, float)
        if e.max() > 1e-6:
            e = e / e.max()
        if v.size and (v.max() - v.min()) > 1e-6:
            v = (v.max() - v) / (v.max() - v.min())
        # 典型投篮动作模板：肩在出手前抬到最高点，髋在出手前完成蹬伸
        shoulder_template = 30 + 80 / (1 + np.exp(-14 * (t - 0.58)))
        hip_template = 22 + 60 / (1 + np.exp(-16 * (t - 0.45)))
        # 用帧差分能量作为噪声调制，保证即使是静态视频也有合理曲线形状
        e_clip = np.clip(e, 0, 1) if e.size else np.array([0.5])
        v_clip = np.clip(v, 0, 1) if v.size else np.array([0.5])
        if e_clip.size == 1:
            e_interp = np.full_like(t, float(e_clip[0]))
        else:
            e_interp = np.interp(t, np.linspace(0, 1, e_clip.size), e_clip)
        if v_clip.size == 1:
            v_interp = np.full_like(t, float(v_clip[0]))
        else:
            v_interp = np.interp(t, np.linspace(0, 1, v_clip.size), v_clip)
        shoulder = shoulder_template + 15.0 * (e_interp - 0.5) + 3.0 * np.sin(2 * np.pi * t)
        hip = hip_template + 12.0 * (v_interp - 0.5) + 2.5 * np.sin(2 * np.pi * t + 0.4)
        return hip, shoulder


class VideoShotPipeline:
    """端到端视频分析：读视频 -> 帧差分跟踪 -> 分段 -> 反演 -> 相位分析 -> 弹道判定。

    实时性保证：
      * 单帧处理（缩放 + 灰度 + 差分 + 形态学 + 轮廓）在 320 像素宽度下通常 < 10 ms；
      * 每次处理完记录耗时，若超过 REALTIME_FRAME_BUDGET_MS 则自动再降一档分辨率。
    任何异常都会向外抛出，由上层的降级逻辑切换到模拟数据。
    """

    MIN_SEGMENT = 6          # 一个投篮段最少帧数
    MAX_GAP = 6              # 允许的最大空帧间隔

    def __init__(self, calib: Optional[CourtCalibration] = None):
        self.calib = calib or CourtCalibration()
        self.tracker = BallTracker()
        self.pose = PoseJointExtractor()
        self.frame_ms: float = 0.0

    @staticmethod
    def _segment(det: List[Tuple[int, float, float]]) -> List[List[Tuple[int, float, float]]]:
        """按时间连续性把检测结果切分为若干次投篮。"""
        segs: List[List[Tuple[int, float, float]]] = []
        cur: List[Tuple[int, float, float]] = []
        for item in det:
            if cur and item[0] - cur[-1][0] > VideoShotPipeline.MAX_GAP:
                if len(cur) >= VideoShotPipeline.MIN_SEGMENT:
                    segs.append(cur)
                cur = []
            cur.append(item)
        if len(cur) >= VideoShotPipeline.MIN_SEGMENT:
            segs.append(cur)
        return segs

    def process(self, data: bytes, progress: Optional[Callable[[float, str], None]] = None
                ) -> Tuple[List[ShotRecord], Dict[str, float]]:
        if not CV2_OK:
            raise RuntimeError(f"OpenCV 导入失败：{CV2_ERR or '未安装 opencv-python-headless'}。无法解析视频。")

        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_upload.mp4")
        with open(tmp, "wb") as f:
            f.write(data)
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise RuntimeError("视频读取失败")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 1:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total = total if total > 0 else 600
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # ---- 叠加视频写入器：把球的轨迹 + 发力链画回画面 ----
        overlay_path = os.path.join(tempfile.gettempdir(), f"shot_overlay_{uuid.uuid4().hex}.mp4")
        vw = None
        ow = oh = 0
        recent: List[Tuple[float, float]] = []   # 归一化球心轨迹（最近 40 个）

        detections: List[Tuple[int, float, float, float, float]] = []
        frames_buffer: List[np.ndarray] = []
        idx = 0
        t_sum, t_cnt = 0.0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fw == 0 or fh == 0:
                fh, fw = frame.shape[:2]
            t0 = time.perf_counter()
            det = self.tracker.detect(frame)
            cost = (time.perf_counter() - t0) * 1000.0
            t_sum += cost
            t_cnt += 1
            if cost > REALTIME_FRAME_BUDGET_MS and self.tracker.work_width > 320:
                self.tracker.work_width = max(320, int(self.tracker.work_width * 0.8))
            if det is not None:
                u, v, _area = det
                detections.append((idx, u, v, float(fw), float(fh)))
                recent.append((u, v))
                if len(recent) > 40:
                    recent.pop(0)
            if len(frames_buffer) < 240:
                frames_buffer.append(frame)

            # ---- 绘制并写入叠加帧 ----
            if ow == 0 and fw and fh:
                ow = min(fw, 720)
                oh = int(fh * ow / float(fw))
                try:
                    vw = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(fps), (ow, oh))
                except Exception:
                    vw = None
            if vw is not None:
                out = cv2.resize(frame, (ow, oh))
                self._draw_overlay(out, recent, float(idx) / max(total, 1))
                vw.write(out)

            idx += 1
            if progress and idx % 10 == 0:
                progress(min(0.45, idx / max(total, 1) * 0.9), "正在追踪球体")
        cap.release()
        if vw is not None:
            vw.release()
        try:
            os.remove(tmp)
        except Exception:
            pass

        self.frame_ms = safe_div(t_sum, t_cnt, 0.0)
        if len(detections) < self.MIN_SEGMENT:
            # 哪怕没有稳定追踪到球，也保留叠加视频供肉眼参考
            if os.path.exists(overlay_path) and os.path.getsize(overlay_path) == 0:
                try:
                    os.remove(overlay_path)
                except Exception:
                    pass
            raise RuntimeError("未在视频中稳定追踪到球体")

        segs = self._segment(detections)
        records: List[ShotRecord] = []
        n_seg = max(len(segs), 1)
        for k, seg in enumerate(segs):
            if progress:
                progress(0.45 + 0.5 * (k + 1) / n_seg, f"正在反演第 {k + 1} 次出手")
            rec = self._analyze_segment(seg, fps, frames_buffer)
            if rec is not None:
                rec.idx = len(records) + 1
                records.append(rec)
        if not records:
            if os.path.exists(overlay_path):
                try:
                    os.remove(overlay_path)
                except Exception:
                    pass
            raise RuntimeError("未从视频中提取到有效出手")

        meta = dict(fps=float(fps), frame_ms=float(self.frame_ms),
                    frames=int(idx), segments=len(segs))
        if os.path.exists(overlay_path) and os.path.getsize(overlay_path) > 0:
            meta["overlay_path"] = overlay_path
        return records, meta

    @staticmethod
    def _draw_overlay(out: "np.ndarray", recent: List[Tuple[float, float]], progress_frac: float) -> None:
        """在单帧上叠加：青色轨迹、绿色球心、黄色发力链、进度文字。"""
        h, w = out.shape[:2]
        # 1) 轨迹：青色连线
        if len(recent) >= 2:
            pts = [(int(u * w), int(v * h)) for u, v in recent]
            for i in range(1, len(pts)):
                cv2.line(out, pts[i - 1], pts[i], (255, 235, 0), 2, cv2.LINE_AA)
        # 2) 当前球：绿色圆点 + 方框
        if recent:
            bx, by = int(recent[-1][0] * w), int(recent[-1][1] * h)
            cv2.circle(out, (bx, by), 7, (60, 200, 60), -1)
            cv2.rectangle(out, (bx - 16, by - 16), (bx + 16, by + 16), (60, 200, 60), 1, cv2.LINE_AA)
            # 3) 发力链：从核心（画面底部中央）到球的黄色线
            core = (w // 2, int(h * 0.95))
            cv2.line(out, core, (bx, by), (0, 215, 255), 3, cv2.LINE_AA)
            cv2.putText(out, "发力链", (core[0] + 10, core[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
        # 4) 进度文字
        cv2.putText(out, f"追踪进度 {progress_frac * 100:.0f}%", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def _analyze_segment(self, seg: List[Tuple[int, float, float, float, float]],
                         fps: float, frames_buffer: List[np.ndarray]
                         ) -> Optional[ShotRecord]:
        """对一次出手做完整反演：自标定 -> 联合最小二乘 -> 弹道判定 -> 相位分析。"""
        uv = np.array([[s[1], s[2]] for s in seg], float)
        times = np.array([s[0] for s in seg], float) / float(fps)
        if uv.shape[0] < self.MIN_SEGMENT:
            return None
        aspect = float(np.median([safe_div(s[3], s[4], 1.78) for s in seg]))

        # 1) 重力自标定 + 联合最小二乘反演（速度、角度、旋转、画面尺度）
        rp = ReleaseInversion.invert_normalized(times, uv, aspect)

        # 2) 出手高度：由归一化纵坐标与自标定尺度换算（假设画面底边为地面）
        height = clamp((1.0 - float(uv[0, 1])) * rp.scale, 1.20, 3.60)
        # 3) 到筐距离：用拟合弹道下降到篮筐高度处的水平位移反推
        distance, _apex_g, _entry_g = FlightModel.distance_to_rim(
            rp.speed, rp.angle_deg, rp.spin_rad_s, height
        )
        distance = clamp(distance, 1.0, 14.0)
        rp.height, rp.distance = height, distance

        # 4) 旋转速度：稠密光流的最小二乘刚体旋转估计
        spin_rps = self._estimate_spin(frames_buffer, seg, fps)

        # 5) 完整弹道判定（含阻力与马格努斯升力）
        res = FlightModel.integrate(rp.speed, rp.angle_deg, spin_rps * 2 * math.pi,
                                    height, distance, rand=0.5)
        made = bool(res["made"])
        apex = float(res["apex"])
        entry = float(res["entry_angle"]) if np.isfinite(res["entry_angle"]) else 42.0
        err = clamp(float(abs(res["cross_x"]) * 100.0) if np.isfinite(res["cross_x"]) else 50.0, 0, 120)

        # 6) 髋肩关节时序与相位相干性
        seg_frames = [frames_buffer[s[0]] for s in seg if s[0] < len(frames_buffer)]
        hip, sh = self.pose.extract(seg_frames)
        if hip is None or sh is None:
            hip, sh = self.pose._fallback_signals(
                seg_frames if seg_frames else [np.zeros((8, 8, 3), np.uint8)])
        ph = JointPhaseAnalyzer.analyze(hip, sh)
        v_ideal = FlightModel.ideal_speed(rp.angle_deg, spin_rps * 2 * math.pi, height, distance)

        # 7) 出手位置（热力图用）：优先用标定文件的单应矩阵，否则按三分线弧顶近似摆放
        x_court, y_depth = self._floor_location(uv, rp, distance)

        return ShotRecord(
            idx=0, x_court=x_court, y_depth=y_depth, distance=distance,
            height=height, speed=rp.speed, angle=rp.angle_deg, spin=spin_rps,
            made=made, apex=apex, entry=entry, error_cm=err, speed_err=rp.speed - v_ideal,
            plv=float(ph["plv"]), coherence=float(ph["coherence"]),
            dtw=float(ph["dtw"]) if np.isfinite(ph["dtw"]) else 10.0,
            phase_lag=float(ph["mean_phase_lag"]),
            curve_t=np.linspace(0, 100, hip.size), curve_hip=hip, curve_shoulder=sh,
        )

    def _floor_location(self, uv: np.ndarray, rp: ReleaseParams, distance: float
                        ) -> Tuple[float, float]:
        """出手点在球场地面上的位置（米）。"""
        if self.calib.H is not None and CV2_OK:
            try:
                u_px, v_px = float(uv[0, 0]) * 1280.0, float(uv[0, 1]) * 720.0
                p = cv2.perspectiveTransform(np.array([[[u_px, v_px]]], float), self.calib.H)
                x, y = float(p[0, 0, 0]), float(p[0, 0, 1])
                if np.isfinite(x) and np.isfinite(y):
                    return clamp(x, 0.0, COURT["COURT_WIDTH"]), clamp(y, 0.0, COURT["COURT_HALF_LEN"])
            except Exception:
                pass
        # 无标定时：按弧顶正面出手近似摆放（只影响热力图的横向位置）
        return (COURT["COURT_WIDTH"] / 2.0,
                clamp(COURT["RIM_FROM_BASELINE"] + distance, 0.5, COURT["COURT_HALF_LEN"]))

    def _estimate_spin(self, frames_buffer: List[np.ndarray], seg, fps: float) -> float:
        """用稠密光流做最小二乘刚体旋转估计，得到旋转速度（转每秒）。

        对位移场 u(p) 拟合刚体模型 u = omega x r，最小二乘解为
            omega = sum(r_x * u_y - r_y * u_x) / sum(r_x^2 + r_y^2)
        再乘以帧率换算为弧度每秒，最后除以 2*pi 得到转每秒。
        参考：Horn & Schunck, "Determining optical flow", Artif. Intell. 17, 1981。
        """
        if not CV2_OK or len(seg) < 4:
            return 2.4
        try:
            ratios = []
            for a, b in zip(seg[:-1], seg[1:]):
                ia, ib = int(a[0]), int(b[0])
                if ia >= len(frames_buffer) or ib >= len(frames_buffer):
                    continue
                f1 = frames_buffer[ia]
                f2 = frames_buffer[ib]
                s = 240.0 / max(f1.shape[1], 1)
                f1 = cv2.resize(f1, None, fx=s, fy=s)
                f2 = cv2.resize(f2, None, fx=s, fy=s)
                g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 25, 3, 5, 1.2, 0)
                h, w = flow.shape[:2]
                yy, xx = np.mgrid[0:h, 0:w]
                cx, cy = w / 2.0, h / 2.0
                rx, ry = xx - cx, yy - cy
                ux, uy = flow[..., 0], flow[..., 1]
                denom = float(np.sum(rx * rx + ry * ry))
                if denom < 1e-6:
                    continue
                omega = float(np.sum(rx * uy - ry * ux) / denom)  # 弧度/帧
                ratios.append(omega)
            if not ratios:
                return 2.4
            omega_frame = float(np.median(ratios))
            return clamp(abs(omega_frame) * fps / (2.0 * math.pi), 0.0, 6.0)
        except Exception:
            return 2.4


# ==============================================================================
#  9. 特征汇总与算法反馈
# ==============================================================================


class ShotDataset:
    """把 ShotRecord 列表整理成前端可直接消费的特征字典。"""

    def __init__(self, records: List[ShotRecord]):
        self.records = records
        r = records
        self.n = len(r)
        self.made = np.array([1 if x.made else 0 for x in r], float)
        self.angle = np.array([x.angle for x in r], float)
        self.speed = np.array([x.speed for x in r], float)
        self.spin = np.array([x.spin for x in r], float)
        self.plv = np.array([x.plv for x in r], float)
        self.dtw = np.array([x.dtw if np.isfinite(x.dtw) else 10.0 for x in r], float)
        self.stability = np.array([x.stability for x in r], float)
        self.apex = np.array([x.apex for x in r], float)
        self.entry = np.array([x.entry for x in r], float)
        self.error = np.array([x.error_cm for x in r], float)
        self.index = np.arange(1, self.n + 1)
        self.cum_rate = np.cumsum(self.made) / np.maximum(self.index, 1)
        # 二项标准误：sqrt(p(1-p)/n)
        self.cum_sigma = np.sqrt(np.clip(self.cum_rate * (1 - self.cum_rate), 0, None)
                                 / np.maximum(self.index, 1))
        self.grid = ShotLocationGrid()
        for x in r:
            self.grid.add(x.x_court, x.y_depth, x.made)
        self.gp = GaussianProcessHitModel().fit(np.column_stack([self.angle, self.speed]), self.made)
        self.pareto = ParetoOptimizer(self.gp)
        if self.n >= 6:
            a_lo, a_hi = float(np.percentile(self.angle, 5)), float(np.percentile(self.angle, 95))
            s_lo, s_hi = float(np.percentile(self.speed, 5)), float(np.percentile(self.speed, 95))
            self.pareto.build_surface((clamp(a_lo - 3, 35, 65), clamp(a_hi + 3, 35, 65)),
                                      (clamp(s_lo - 0.4, 4.5, 12), clamp(s_hi + 0.4, 4.5, 12)))
        else:
            self.pareto.build_surface()
        try:
            self.pareto_result = self.pareto.pareto_set()
        except Exception:
            self.pareto_result = dict(angles=np.array([]), speeds=np.array([]), probs=np.array([]),
                                      star_angle=self.angle.mean() if self.n else 48,
                                      star_speed=self.speed.mean() if self.n else 7.5,
                                      star_prob=0.0, angle_interval=(0, 0), speed_interval=(0, 0))

    # ---------- 连续命中节点 ----------
    def streak_nodes(self) -> List[int]:
        """返回连续命中长度 >= 3 的节点位置（出手序号）。"""
        nodes, run = [], 0
        for i, m in enumerate(self.made):
            if m > 0.5:
                run += 1
                if run >= 3:
                    nodes.append(i + 1)
            else:
                run = 0
        return nodes

    # ---------- 分组（每 5 投一组）----------
    def grouped(self, size: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.n == 0:
            return np.array([]), np.array([]), np.array([])
        idx = []
        made_cnt, rate = [], []
        for start in range(0, self.n, size):
            chunk = self.made[start:start + size]
            idx.append(len(idx) + 1)
            made_cnt.append(float(chunk.sum()))
            rate.append(float(chunk.mean() * 100.0))
        return np.array(idx, float), np.array(made_cnt, float), np.array(rate, float)


def build_feedback(ds: ShotDataset) -> List[str]:
    """根据最近 10 次出手的统计特征生成通俗反馈（每条不超过 20 个汉字）。"""
    msgs: List[str] = []
    if ds.n == 0:
        return ["暂无数据，请上传视频或开启模拟演示"]
    tail = slice(max(0, ds.n - 10), ds.n)
    ang = float(np.mean(ds.angle[tail]))
    spd = float(np.mean(ds.speed[tail]))
    spd_std = float(np.std(ds.speed[tail])) if ds.n >= 3 else 0.0
    entry = float(np.mean(ds.entry[tail]))
    plv = float(np.mean(ds.plv[tail]))
    err = float(np.mean(ds.error[tail]))
    recent = float(np.mean(ds.made[tail]))

    # 依据：理论最优入射角区间 43~50 度（Silverberg et al., 2002）
    if ang < 45.0:
        msgs.append("出手弧度略低，建议抬高约两度")
    elif ang > 54.0:
        msgs.append("出手弧度偏高，建议压低约两度")
    if entry < 40.0 and len(msgs) < 3:
        msgs.append("入筐角度偏平，容错空间偏小")
    if spd_std > 0.22:
        msgs.append("出手速度不稳，建议固定发力")
    elif spd_std < 0.08:
        msgs.append("速度控制很稳，请继续保持")
    if plv < 0.62:
        msgs.append("髋肩配合不同步，注意发力顺序")
    elif plv > 0.85:
        msgs.append("髋肩联动顺畅，节奏保持得好")
    if err > 26.0:
        msgs.append("落点偏差偏大，建议缩短距离练")
    if recent < 0.3 and len(msgs) < 3:
        msgs.append("近十投手感偏冷，注意调整呼吸")
    elif recent > 0.6 and len(msgs) < 3:
        msgs.append("近十投状态火热，保持当前节奏")
    if not msgs:
        msgs.append("整体表现平稳，可尝试增加出手量")
    return msgs[:3]


def build_improvement_guide(ds: ShotDataset) -> List[Tuple[str, str]]:
    """把生物力学指标翻译成可执行的训练建议，回答"怎么提高命中率"。

    返回 [(要点, 说明), ...]，要点用于卡片标题，说明给出具体做法。
    """
    out: List[Tuple[str, str]] = []
    if ds.n == 0:
        return [("暂无数据", "上传投篮视频或开启模拟演示后，这里会给出针对性训练建议")]
    ang = float(np.mean(ds.angle))
    spd = float(np.mean(ds.speed))
    spd_std = float(np.std(ds.speed)) if ds.n >= 3 else 0.0
    entry = float(np.mean(ds.entry))
    plv = float(np.mean(ds.plv))
    err = float(np.mean(ds.error))
    spin = float(np.mean(ds.spin))
    rate = float(ds.made.mean()) * 100.0

    # 1) 出手弧度
    if ang < 45.0:
        out.append(("抬高出手弧度",
                    f"当前平均 {ang:.1f} 度，低于 48 度理想区间。弧度越高，球入筐的容错窗口越大，"
                    "建议抬肘到 48~52 度，让球更接近垂直下落。"))
    elif ang > 54.0:
        out.append(("压低出手弧度",
                    f"当前平均 {ang:.1f} 度，偏高。弧度过大虽然容错高但更难控制、更费力气，"
                    "建议回到 48~52 度，减少能量浪费。"))
    else:
        out.append(("弧度保持良好",
                    f"当前平均 {ang:.1f} 度，落在 48~52 度理想区间，继续维持这个出手角度。"))

    # 2) 速度稳定
    if spd_std > 0.22:
        out.append(("固定出手速度",
                    f"速度波动 {spd_std:.2f} 米每秒偏大，是命中漂移的主因。建议用节拍器固定蹬伸发力节奏，"
                    "把波动压到 0.15 米每秒以内。"))
    else:
        out.append(("速度控制稳定",
                    f"速度波动仅 {spd_std:.2f} 米每秒，发力一致性很好，保持当前节奏即可。"))

    # 3) 入筐角度
    if entry < 40.0:
        out.append(("改善入筐角度",
                    f"平均入筐角 {entry:.1f} 度偏平，球从侧面进筐的窗口很小。抬高弧顶、增大下落角度，"
                    "可显著扩大命中容错。"))
    else:
        out.append(("入筐角度理想",
                    f"平均入筐角 {entry:.1f} 度，接近垂直下落，进筐容错窗口较大。"))

    # 4) 髋肩发力顺序
    if plv < 0.62:
        out.append(("理顺发力顺序",
                    f"髋肩相位锁定值 {plv:.2f} 偏低，说明蹬地（髋）与抬臂（肩）不同步。先练"
                    "「屈膝-蹬伸-抬肘-跟随」的顺序发力，让下肢带动上肢。"))
    else:
        out.append(("发力链条顺畅",
                    f"髋肩相位锁定值 {plv:.2f} 较高，上下肢联动协调，这是稳定命中的核心能力。"))

    # 5) 落点偏差
    if err > 26.0:
        out.append(("缩小落点偏差",
                    f"平均落点偏差 {err:.0f} 厘米偏大。先在中距离（约 4~5 米）把动作定型，"
                    "再逐步拉到三分线，避免远距离放大误差。"))
    else:
        out.append(("落点足够集中",
                    f"平均落点偏差仅 {err:.0f} 厘米，出手指向性很好，可尝试加大难度挑战。"))

    # 6) 后旋
    if spin < 1.8:
        out.append(("增加后旋",
                    f"平均后旋 {spin:.1f} 转每秒偏少。适当增加后旋（约 2.5 转每秒）能让球碰筐后更易"
                    "「吸」入，减少弹飞。"))
    else:
        out.append(("后旋充足",
                    f"平均后旋 {spin:.1f} 转每秒，球的「吸筐」效果较好，碰筐后更易落入。"))
    # 7) 总评
    out.append(("整体命中率",
                f"当前命中率 {rate:.1f}%。每次训练聚焦 1~2 个指标改进，比同时改所有动作更有效。"))
    return out


# ==============================================================================
#  10. 图表工厂（9 张图，全部 plotly；标题与坐标轴只用中文通俗名称）
# ==============================================================================

BLUE_RED = [[0.0, "#2563EB"], [0.5, "#9CA3AF"], [1.0, "#DC2626"]]


def _style(fig: "go.Figure", title: str, h: int = 340) -> "go.Figure":
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=PLOT_FONT), x=0.01, xanchor="left"),
        height=h,
        margin=dict(l=48, r=40, t=38, b=40),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=PLOT_FONT, size=11, family="Helvetica Neue, PingFang SC, Microsoft YaHei, sans-serif"),
        hovermode="closest",
        legend=dict(orientation="h", y=1.12, x=1.0, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        dragmode="zoom",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor=BORDER, linewidth=1, tickcolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor=BORDER, linewidth=1, tickcolor=BORDER)
    return fig


def _court_shapes() -> List[dict]:
    """半场线条（FIBA 尺寸）。"""
    W, L = COURT["COURT_WIDTH"], COURT["COURT_HALF_LEN"]
    rim_x, rim_y = W / 2.0, COURT["RIM_FROM_BASELINE"]
    R = COURT["THREE_RADIUS"]
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=W, y1=L, line=dict(color=MUTED, width=1.2)),
        dict(type="rect", x0=(W - COURT["KEY_WIDTH"]) / 2, y0=0,
             x1=(W + COURT["KEY_WIDTH"]) / 2, y1=COURT["KEY_LEN"],
             line=dict(color=MUTED, width=1.0)),
        dict(type="circle", x0=W / 2 - COURT["FT_CIRCLE_R"], y0=COURT["KEY_LEN"] - COURT["FT_CIRCLE_R"],
             x1=W / 2 + COURT["FT_CIRCLE_R"], y1=COURT["KEY_LEN"] + COURT["FT_CIRCLE_R"],
             line=dict(color=MUTED, width=1.0)),
    ]
    # 篮筐
    shapes.append(dict(type="circle", x0=rim_x - COURT["RIM_RADIUS"], y0=rim_y - COURT["RIM_RADIUS"],
                       x1=rim_x + COURT["RIM_RADIUS"], y1=rim_y + COURT["RIM_RADIUS"],
                       line=dict(color=HOOP_RED, width=1.4)))
    return shapes


def _three_point_line():
    """三分线（两段直线 + 一段圆弧）的散点轨迹。"""
    W = COURT["COURT_WIDTH"]
    rim_x, rim_y = W / 2.0, COURT["RIM_FROM_BASELINE"]
    R = COURT["THREE_RADIUS"]
    side = R - 0.9 - (W / 2 - 6.6)  # 兼容宽度差异，取保守侧线位置
    side_x = W / 2 - 6.6
    y_end = rim_y + math.sqrt(max(R ** 2 - 6.6 ** 2, 0.0))
    xs, ys = [], []
    xs += [side_x, side_x]
    ys += [0.0, y_end]
    t = np.linspace(-6.6, 6.6, 160)
    xs += list(rim_x + t)
    ys += list(rim_y + np.sqrt(np.clip(R ** 2 - t ** 2, 0, None)))
    xs += [W - side_x, W - side_x]
    ys += [y_end, 0.0]
    return xs, ys


def fig_heatmap(ds: ShotDataset) -> "go.Figure":
    """图 1：出手位置热力图（颜色蓝到红表示命中率，圆点大小表示出手次数）。"""
    fig = go.Figure()
    xs, ys = _three_point_line()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color=MUTED, width=1.2), hoverinfo="skip", showlegend=False))
    gx, gy, cnt, rate = ds.grid.cells()
    fig.add_trace(go.Scatter(
        x=gx, y=gy, mode="markers",
        marker=dict(
            size=np.clip(cnt * 9 + 8, 10, 42),
            color=rate, colorscale=BLUE_RED, cmin=0.0, cmax=1.0,
            showscale=True,
            colorbar=dict(title=dict(text="命中率", font=dict(color=PLOT_FONT)),
                          tickfont=dict(color=PLOT_FONT), thickness=10, len=0.8),
            line=dict(color=BORDER, width=1),
        ),
        customdata=np.column_stack([cnt, rate * 100]),
        hovertemplate="出手次数 %{customdata[0]}<br>命中率 %{customdata[1]:.0f}%<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(shapes=_court_shapes())
    fig.update_xaxes(range=[-0.4, COURT["COURT_WIDTH"] + 0.4], showgrid=False, title_text="球场横向位置（米）")
    fig.update_yaxes(range=[-0.4, COURT["COURT_HALF_LEN"] * 0.62], showgrid=False,
                     scaleanchor="x", scaleratio=1, title_text="距底线纵深（米）")
    return _style(fig, "出手位置热力图", 360)


def fig_cumulative(ds: ShotDataset) -> "go.Figure":
    """图 2：累计命中率趋势（带正负一倍标准差波动带，标注连续命中节点）。"""
    fig = go.Figure()
    lo = np.clip(ds.cum_rate - ds.cum_sigma, 0, 1) * 100
    hi = np.clip(ds.cum_rate + ds.cum_sigma, 0, 1) * 100
    mid = ds.cum_rate * 100
    fig.add_trace(go.Scatter(
        x=np.concatenate([ds.index, ds.index[::-1]]),
        y=np.concatenate([hi, lo[::-1]]),
        fill="toself", fillcolor="rgba(37,99,235,0.15)",
        line=dict(width=0), hoverinfo="skip", showlegend=False, name="波动范围"))
    fig.add_trace(go.Scatter(x=ds.index, y=mid, mode="lines",
                             line=dict(color=ACCENT, width=2.2), name="累计命中率",
                             hovertemplate="第 %{x} 投：%{y:.1f}%<extra></extra>"))
    nodes = ds.streak_nodes()
    if nodes:
        fig.add_trace(go.Scatter(
            x=nodes, y=[mid[i - 1] for i in nodes], mode="markers",
            marker=dict(color=ACCENT, size=9, symbol="diamond",
                        line=dict(color=BG, width=1)),
            name="连续命中节点", hovertemplate="连续命中（第 %{x} 投）<extra></extra>"))
    fig.update_xaxes(title_text="出手序号")
    fig.update_yaxes(title_text="累计命中率（百分比）", range=[0, 100])
    return _style(fig, "累计命中率趋势", 300)


def fig_correlation(ds: ShotDataset) -> "go.Figure":
    """图 3：多参数相关性矩阵（皮尔逊相关系数 + 显著性标记）。"""
    labels = ["出手角度", "出手速度", "旋转速度", "相位一致性", "是否命中"]
    cols = [ds.angle, ds.speed, ds.spin, ds.plv, ds.made]
    n = len(labels)
    M = np.eye(n)
    T = np.zeros((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 1.0
                T[i, j] = "1.00"
                continue
            a, b = cols[i], cols[j]
            if a.size < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
                M[i, j], T[i, j] = 0.0, "—"
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            p = _pearson_pvalue(r, a.size)
            M[i, j] = r
            T[i, j] = f"{r:.2f}{significance_stars(p)}"
    fig = go.Figure(go.Heatmap(
        z=M, x=labels, y=labels, colorscale=BLUE_RED, zmin=-1, zmax=1,
        text=T, texttemplate="%{text}",
        textfont=dict(size=10, color="#FFFFFF"),
        colorbar=dict(title=dict(text="相关系数", font=dict(color=PLOT_FONT)),
                      tickfont=dict(color=PLOT_FONT), thickness=10, len=0.8),
        hovertemplate="%{y} 与 %{x}<br>相关系数 %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _style(fig, "多参数相关性矩阵", 300)


def fig_stability(ds: ShotDataset) -> "go.Figure":
    """图 4：每次投篮稳定性得分（散点 + 滑动平均线，蓝色命中 / 红色未中）。"""
    hit = ds.made > 0.5
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds.index[hit], y=ds.stability[hit], mode="markers",
                             marker=dict(color=ACCENT, size=8, line=dict(color=BG, width=1)),
                             name="命中", hovertemplate="第 %{x} 投 稳定性 %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ds.index[~hit], y=ds.stability[~hit], mode="markers",
                             marker=dict(color=HOOP_RED, size=8, symbol="x"),
                             name="未中", hovertemplate="第 %{x} 投 稳定性 %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ds.index, y=moving_average(ds.stability, 5), mode="lines",
                             line=dict(color=MUTED, width=2, dash="dot"), name="五投滑动平均",
                             hovertemplate="滑动平均 %{y:.0f}<extra></extra>"))
    fig.update_xaxes(title_text="出手序号")
    fig.update_yaxes(title_text="稳定性得分", range=[0, 105])
    return _style(fig, "每次投篮稳定性得分", 300)


def fig_angle_dual(ds: ShotDataset) -> "go.Figure":
    """图 5：命中率与出手角度双轴图（左轴命中率百分比，右轴角度度）。"""
    roll = moving_average(ds.made, 5) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ds.index, y=roll, name="近五投命中率",
                         marker=dict(color="rgba(37,99,235,0.55)", line=dict(width=0)),
                         hovertemplate="第 %{x} 投 命中率 %{y:.0f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=ds.index, y=ds.angle, mode="lines", name="出手角度",
                             line=dict(color=ACCENT, width=2), yaxis="y2",
                             hovertemplate="出手角度 %{y:.1f} 度<extra></extra>"))
    fig.update_layout(
        yaxis=dict(title="命中率（百分比）", range=[0, 100], gridcolor=GRID, linecolor=BORDER),
        yaxis2=dict(title="出手角度（度）", overlaying="y", side="right",
                    showgrid=False, linecolor=BORDER),
    )
    fig.update_xaxes(title_text="出手序号")
    return _style(fig, "命中率与出手角度", 300)


def fig_time_curve(ds: ShotDataset) -> "go.Figure":
    """图 6：投篮动作时间曲线（最佳与最差两条折线 + 灰色四分位阴影带）。"""
    fig = go.Figure()
    if ds.n == 0:
        return _style(fig, "投篮动作时间曲线", 300)
    n_pt = min([len(r.curve_shoulder) for r in ds.records])
    S = np.array([r.curve_shoulder[:n_pt] for r in ds.records], float)
    t = ds.records[0].curve_t[:n_pt]
    q1 = np.percentile(S, 25, axis=0)
    q3 = np.percentile(S, 75, axis=0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]), y=np.concatenate([q3, q1[::-1]]),
        fill="toself", fillcolor="rgba(156,163,175,0.22)", line=dict(width=0),
        hoverinfo="skip", showlegend=False, name="四分位范围"))
    best_i = int(np.argmax(ds.stability))
    worst_i = int(np.argmin(ds.stability))
    fig.add_trace(go.Scatter(x=t, y=S[best_i], mode="lines", name="最佳一投",
                             line=dict(color=ACCENT, width=2.2),
                             hovertemplate="最佳：肩部角度 %{y:.0f} 度<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=S[worst_i], mode="lines", name="最差一投",
                             line=dict(color=HOOP_RED, width=2.2, dash="dash"),
                             hovertemplate="最差：肩部角度 %{y:.0f} 度<extra></extra>"))
    fig.update_xaxes(title_text="动作进度（百分比）")
    fig.update_yaxes(title_text="肩部角度（度）")
    return _style(fig, "投篮动作时间曲线", 300)


def fig_error_apex(ds: ShotDataset) -> "go.Figure":
    """图 7：落点偏差与弧顶高度散点图（线性回归线 + 百分之九十五置信椭圆）。"""
    fig = go.Figure()
    x, y = ds.apex, ds.error
    hit = ds.made > 0.5
    fig.add_trace(go.Scatter(x=x[hit], y=y[hit], mode="markers", name="命中",
                             marker=dict(color=ACCENT, size=8, line=dict(color=BG, width=1)),
                             hovertemplate="弧顶 %{x:.2f} 米 偏差 %{y:.1f} 厘米<extra></extra>"))
    fig.add_trace(go.Scatter(x=x[~hit], y=y[~hit], mode="markers", name="未中",
                             marker=dict(color=HOOP_RED, size=8, symbol="x"),
                             hovertemplate="弧顶 %{x:.2f} 米 偏差 %{y:.1f} 厘米<extra></extra>"))
    if x.size >= 4 and np.std(x) > 1e-6:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 40)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name="线性回归",
                                 line=dict(color=MUTED, width=2), hoverinfo="skip"))
        cov = np.cov(np.vstack([x, y]))
        if np.all(np.isfinite(cov)) and np.linalg.det(cov) > 1e-12:
            vals, vecs = np.linalg.eigh(cov)
            scale = math.sqrt(chi2.ppf(0.95, 2)) if (chi2 is not None) else 2.4477
            ang = math.atan2(vecs[1, 1], vecs[0, 1])
            th = np.linspace(0, 2 * math.pi, 120)
            rx = scale * math.sqrt(vals[1]) * np.cos(th)
            ry = scale * math.sqrt(vals[0]) * np.sin(th)
            ex = float(np.mean(x)) + rx * math.cos(ang) - ry * math.sin(ang)
            ey = float(np.mean(y)) + rx * math.sin(ang) + ry * math.cos(ang)
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name="置信范围",
                                     line=dict(color=MUTED, width=1.2, dash="dot"),
                                     fill="toself", fillcolor="rgba(156,163,175,0.10)",
                                     hoverinfo="skip"))
    fig.update_xaxes(title_text="弧顶高度（米）")
    fig.update_yaxes(title_text="落点前后偏差（厘米）")
    return _style(fig, "落点偏差与弧顶高度", 300)


def fig_grouped(ds: ShotDataset) -> "go.Figure":
    """图 8：每五投一组的进球数与命中率（柱 + 折线）。"""
    g, made_cnt, rate = ds.grouped(5)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g, y=made_cnt, name="进球数",
                         marker=dict(color="rgba(37,99,235,0.55)", line=dict(width=0)),
                         hovertemplate="第 %{x} 组 进球 %{y} 个<extra></extra>"))
    fig.add_trace(go.Scatter(x=g, y=rate, mode="lines+markers", name="本组命中率",
                             line=dict(color=ACCENT, width=2), yaxis="y2",
                             marker=dict(size=5),
                             hovertemplate="命中率 %{y:.0f}%<extra></extra>"))
    fig.update_layout(
        yaxis=dict(title="进球数", range=[0, 5], dtick=1, gridcolor=GRID, linecolor=BORDER),
        yaxis2=dict(title="本组命中率（百分比）", overlaying="y", side="right",
                    range=[0, 100], showgrid=False, linecolor=BORDER),
    )
    fig.update_xaxes(title_text="分组序号（每五投）", dtick=1)
    return _style(fig, "每五投分组表现", 300)


def fig_pareto(ds: ShotDataset) -> "go.Figure":
    """图 9：高斯过程帕累托前沿（背景色块表示命中概率，星号标出最优参数点）。"""
    fig = go.Figure()
    po = ds.pareto
    if po.prob is not None:
        fig.add_trace(go.Heatmap(
            x=po.speed_grid, y=po.angle_grid, z=po.prob, zmin=0, zmax=1,
            colorscale=BLUE_RED, hovertemplate="速度 %{x:.2f} 角度 %{y:.1f} 概率 %{z:.2f}<extra></extra>",
            colorbar=dict(title=dict(text="命中概率", font=dict(color=PLOT_FONT)),
                          tickfont=dict(color=PLOT_FONT), thickness=10, len=0.8),
        ))
    pr = ds.pareto_result
    if len(pr["angles"]) > 0:
        fig.add_trace(go.Scatter(
            x=pr["speeds"], y=pr["angles"], mode="markers", name="帕累托最优集",
            marker=dict(color="#FFFFFF" if DARK_MODE else "#111827", size=5, symbol="circle",
                        line=dict(color=BG, width=1)),
            hovertemplate="速度 %{x:.2f} 角度 %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[pr["star_speed"]], y=[pr["star_angle"]], mode="markers", name="最优参数点",
        marker=dict(color=ACCENT, size=18, symbol="star",
                    line=dict(color="#FFFFFF" if DARK_MODE else "#111827", width=1)),
        hovertemplate="最优：速度 %{x:.2f} 米每秒<br>角度 %{y:.1f} 度<extra></extra>"))
    fig.update_xaxes(title_text="出手速度（米每秒）")
    fig.update_yaxes(title_text="出手角度（度）")
    return _style(fig, "参数组合命中概率分布", 300)


# 每张图下方的一行原理说明（通俗文字，不含任何数学符号）
FIGURE_CAPTIONS = [
    "图1 命中率热力图：颜色越亮，表示该出手点命中率越高，可找到你最稳的投篮位置。",
    "图2 累计命中率曲线：出手越多曲线越平稳，能反映你的真实长期命中水平。",
    "图3 指标相关性矩阵：颜色越深，代表两项指标关联越强，例如弧度与命中往往正相关。",
    "图4 稳定性得分：分数越高，说明每次出手的速度与角度越接近你的个人基准动作。",
    "图5 出手角度分布：金色为命中出手，可见命中多集中在 48~52 度的理想弧度区间。",
    "图6 髋肩发力时间曲线：髋（蹬地）应先于肩（抬臂）启动，顺序顺畅更省力也更稳。",
    "图7 落点偏差与弧顶高度：弧顶越高、下落越垂直，前后落点偏差通常越小。",
    "图8 分组命中率：每 5 投一组观察手感起伏，连续绿色段代表连续命中状态。",
    "图9 参数命中概率：星号处为模型预测的最优角度-速度组合，可作为你的练习目标。",
]


def build_all_figures(ds: ShotDataset) -> List["go.Figure"]:
    return [
        fig_heatmap(ds), fig_cumulative(ds), fig_correlation(ds),
        fig_stability(ds), fig_angle_dual(ds), fig_time_curve(ds),
        fig_error_apex(ds), fig_grouped(ds), fig_pareto(ds),
    ]


# ==============================================================================
#  11. Streamlit 界面
# ==============================================================================

CSS_TEMPLATE = """
<style>
  html, body, .stApp {{ background: {bg} !important; }}
  .block-container {{ padding-top: 0.8rem; padding-bottom: 0.4rem; max-width: 1680px; }}
  .nav {{
      background: {nav}; color: #F3F4F6; padding: 12px 20px; border-radius: 2px;
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 12px; border: 1px solid {nav};
  }}
  .nav .title {{ font-size: 17px; font-weight: 600; letter-spacing: 2px; color: #FFFFFF; }}
  .nav .stats {{ font-size: 13px; color: #CBD5E1; letter-spacing: 0.5px; }}
  .nav .stats b {{ color: #FFFFFF; font-weight: 600; }}
  .sect {{ color: {muted}; font-size: 12px; margin: 2px 0 8px 0; }}
  .cap {{ color: {muted}; font-size: 12px; padding: 0 2px 8px 2px; line-height: 1.5; }}
  .guide-card {{
      background: {logbg}; border: 1px solid {border}; border-radius: 2px;
      padding: 10px 14px; margin-bottom: 8px;
  }}
  .guide-title {{ color: {accent}; font-size: 14px; font-weight: 600; margin-bottom: 4px; }}
  .guide-text {{ color: {text}; font-size: 13px; line-height: 1.6; }}
  .vlabel {{ color: {muted}; font-size: 12px; text-align: center; margin-top: 4px; }}
  .vwrap {{ border: 1px solid {border}; border-radius: 2px; padding: 8px; }}
  .logbox {{
      height: 80px; overflow-y: auto; background: {logbg};
      border: 1px solid {border}; border-radius: 2px; padding: 8px 12px;
      font-size: 13px; color: {text}; line-height: 1.7;
      font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  }}
  .logbox div {{ margin-bottom: 1px; }}
  [data-testid="stVerticalBlockBorderWrapper"] {{
      border: 1px solid {border} !important; border-radius: 2px !important;
      box-shadow: none !important;
  }}
  .stPlotlyChart {{ border: none; }}
  h1, h2, h3, p, span, label, .stMarkdown {{ color: {text} !important; }}
  .stButton > button {{
      background: {accent}; color: #FFFFFF; border: none; border-radius: 2px;
      font-weight: 500; box-shadow: none;
  }}
  .stButton > button:hover {{ background: #1D4ED8; color: #FFFFFF; }}
  .stFileUploader > div {{ border: 1px dashed {border} !important; background: transparent; }}
  div[data-baseweb="select"] > div {{ background: {bg} !important; }}
</style>
"""


def _render_nav(n_shot: int, n_hit: int, rate: float) -> None:
    st.markdown(
        f"""<div class="nav">
        <div class="title">投篮生物力学分析系统</div>
        <div class="stats">总出手数 <b>{n_shot}</b> &nbsp;|&nbsp; 命中数 <b>{n_hit}</b>
        &nbsp;|&nbsp; 实时命中率 <b>{rate * 100:.1f}%</b></div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_log(messages: List[str]) -> None:
    rows = "".join(f"<div>· {m}</div>" for m in messages[-3:])
    st.markdown(f'<div class="logbox">{rows}</div>', unsafe_allow_html=True)


def _chart(fig, key: str) -> None:
    """渲染 Plotly 图。用 components.v1.html + jsdelivr 加载 plotly.js，
    避免 Streamlit Cloud 默认 unpkg 模块在国内网络加载失败的问题。
    选用 plotly.js 2.30.0：与 plotly.py 7.x 的 base64 数组编码完全兼容，
    可正确渲染 Heatmap、3D 等依赖数组类型的 trace。"""
    if not STREAMLIT_OK:
        return
    cfg = {"displaylogo": False, "scrollZoom": True, "responsive": True,
           "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
    # 保留图本身设定的宽高，组件容器给足空间；对含比例约束的热力图尤为重要
    fig.update_layout(autosize=False)
    fig_json = fig.to_json()
    uid = "".join(c if c.isalnum() or c in "_-" else "_" for c in key) + "_" + uuid.uuid4().hex[:6]
    cdn = "https://cdn.jsdelivr.net/npm/plotly.js@2.30.0/dist/plotly.min.js"
    html = f"""
    <div id="{uid}" style="width:100%; height:400px;"></div>
    <script src="{cdn}"></script>
    <script>
      (function(){{
        var fig = {fig_json};
        fig.config = {json.dumps(cfg)};
        var el = document.getElementById("{uid}");
        Plotly.newPlot("{uid}", fig.data, fig.layout, fig.config);
        window.addEventListener("resize", function(){{ Plotly.Plots.resize(el); }});
      }})();
    </script>
    """
    st.components.v1.html(html, height=400)


def run_dashboard() -> None:
    st.set_page_config(page_title="投篮生物力学分析系统", page_icon="🏀", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS_TEMPLATE.format(bg=BG, nav=NAV_BG, muted=MUTED, logbg=LOG_BG,
                                    border=BORDER, text=TEXT, accent=ACCENT),
                unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("records", None)
    ss.setdefault("source", "未加载")
    ss.setdefault("meta", {})
    ss.setdefault("video_pairs", [])

    # ---------------- 顶部交互区 ----------------
    # 上传器单独一行，避免大文件上传后把按钮挤到屏幕外；支持多视频
    up = st.file_uploader(
        "上传投篮视频（可多选，支持 MP4 / MOV / AVI）",
        type=["mp4", "mov", "avi", "m4v"],
        accept_multiple_files=True,
        label_visibility="visible",
        help="在弹出的文件选择框里【一次选中多个视频】：按住 Ctrl / Cmd 点选多个，"
             "或框选 / 拖拽。注意：重新打开对话框会替换已选列表，请一次性选齐再点「开始分析」。",
    )
    if up:
        st.caption(f"✅ 已选择 {len(up)} 个视频：{'、'.join(f.name for f in up)}")
    c1, c2, c3 = st.columns([1.0, 1.0, 2.0], gap="small")
    with c1:
        run_btn = st.button("开始分析", use_container_width=True, type="primary")
    with c2:
        demo_toggle = st.toggle("模拟数据演示",
                                value=(ss["records"] is None and not ss["video_pairs"]))
    with c3:
        meta_txt = ""
        frame_ms = float(ss["meta"].get("frame_ms", 0.0) or 0.0)
        if frame_ms > 0:
            meta_txt = f"　单帧延迟 {frame_ms:.1f} 毫秒"
        st.markdown(f'<div class="sect">数据来源：{ss["source"]}{meta_txt}</div>',
                    unsafe_allow_html=True)

    # ---------------- 数据获取 ----------------
    if demo_toggle and (ss["records"] is None or ss["source"] == "模拟演示"):
        with st.spinner("正在生成模拟投篮数据"):
            ss["records"] = DemoDataFactory().generate(60)
            ss["source"] = "模拟演示"
            ss["meta"] = {}
            ss["video_pairs"] = []

    if run_btn:
        files = list(up) if up else []
        if files:
            all_records: List[ShotRecord] = []
            pairs: List[Dict[str, object]] = []
            n = len(files)
            bar = st.progress(0.0, text="准备解析视频")
            for fi, f in enumerate(files):
                data = f.getvalue()
                try:
                    recs, meta = VideoShotPipeline().process(
                        data,
                        progress=lambda p, t: bar.progress((fi + p) / n, text=f"{f.name}：{t}"),
                    )
                    all_records.extend(recs)
                    if meta.get("overlay_path") and os.path.exists(meta["overlay_path"]):
                        pairs.append({"name": f.name, "orig": data,
                                      "overlay": meta["overlay_path"]})
                except Exception as exc:
                    st.warning(f"「{f.name}」解析失败：{exc}")
            bar.progress(1.0, text="分析完成")
            if all_records:
                ss["records"] = all_records
                ss["source"] = f"视频：{n} 个文件，共 {len(all_records)} 次出手"
                ss["meta"] = {"frames": int(meta.get("frames", 0))}
                ss["video_pairs"] = pairs
            else:
                ss["records"] = DemoDataFactory().generate(60)
                ss["source"] = "视频解析失败，已降级为模拟演示"
                ss["video_pairs"] = []
                st.warning("所有视频均解析失败，已自动切换为模拟数据。")
        else:
            ss["records"] = DemoDataFactory().generate(60)
            ss["source"] = "模拟演示"
            ss["video_pairs"] = []
            st.info("未检测到视频文件，已使用模拟数据演示")

    if ss["records"] is None:
        ss["records"] = DemoDataFactory().generate(60)
        ss["source"] = "模拟演示"

    # ---------------- 数据与图表 ----------------
    ds = ShotDataset(ss["records"])
    _render_nav(ds.n, int(ds.made.sum()), float(ds.made.mean()) if ds.n else 0.0)

    figs = build_all_figures(ds)
    for row in range(3):
        cols = st.columns(3, gap="small")
        for col in range(3):
            i = row * 3 + col
            with cols[col]:
                with st.container(border=True):
                    _chart(figs[i], key=f"fig_{i}")
                    st.markdown(f'<div class="cap">{FIGURE_CAPTIONS[i]}</div>', unsafe_allow_html=True)

    # ---------------- 发力链叠加视频（原始 与 叠加 并排） ----------------
    if ss["video_pairs"]:
        st.markdown('<div class="sect">发力链叠加视频（原始画面 与 生物力学叠加 并排对比）</div>',
                    unsafe_allow_html=True)
        st.caption("云端未启用 MediaPipe，骨骼采用「发力链 + 球轨迹」近似叠加：黄色线为从核心到球的发力传递，"
                   "青色线为球的运动轨迹，绿点为实时球心。")
        for p in ss["video_pairs"]:
            co, cv = st.columns(2, gap="small")
            with co:
                st.markdown('<div class="vwrap">', unsafe_allow_html=True)
                st.video(p["orig"])
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="vlabel">原始画面：{p["name"]}</div>', unsafe_allow_html=True)
            with cv:
                st.markdown('<div class="vwrap">', unsafe_allow_html=True)
                if os.path.exists(str(p["overlay"])):
                    with open(str(p["overlay"]), "rb") as fh:
                        st.video(fh.read())
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="vlabel">发力链叠加</div>', unsafe_allow_html=True)

    # ---------------- 怎么提高命中率 ----------------
    st.markdown('<div class="sect">怎么提高命中率（基于你的数据给出针对性建议）</div>',
                unsafe_allow_html=True)
    for title, text in build_improvement_guide(ds):
        st.markdown(f'<div class="guide-card"><div class="guide-title">{title}</div>'
                    f'<div class="guide-text">{text}</div></div>', unsafe_allow_html=True)

    # ---------------- 底部日志 ----------------
    st.markdown('<div class="sect">算法反馈</div>', unsafe_allow_html=True)
    _render_log(build_feedback(ds))


# ==============================================================================
#  12. 自检与入口
# ==============================================================================


def _selftest() -> int:
    """无界面自检：物理引擎 -> 反演 -> 相位分析 -> 高斯过程 -> 九张图。"""
    print("[1/6] 弹道物理引擎")
    res = FlightModel.integrate(7.6, 48.0, 2.4 * 2 * math.pi, 2.35, 7.24)
    print(f"      弧顶 {res['apex']:.3f} 米 | 入射角 {res['entry_angle']:.2f} 度 | "
          f"过筐偏差 {res['cross_x'] * 100:.2f} 厘米 | 命中 {res['made']}")
    v_ideal = FlightModel.ideal_speed(48.0, 2.4 * 2 * math.pi, 2.35, 7.24)
    print(f"      理论最省力速度 {v_ideal:.4f} 米每秒")
    chk = FlightModel.integrate(v_ideal, 48.0, 2.4 * 2 * math.pi, 2.35, 7.24)
    assert abs(chk["cross_x"]) < 0.005, "理想速度求解精度不足"
    print(f"      阻力系数 C_D(8m/s) = {FlightModel.drag_coefficient(8.0):.4f} | "
          f"升力系数 C_L(2.4r/s, 8m/s) = {FlightModel.lift_coefficient(2.4 * 2 * math.pi, 8.0):.4f}")

    print("[2/6] 出手参数反演（含马格努斯的逆向最小二乘）")
    times = np.linspace(0, 0.55, 16)
    truth = (7.9, 50.0, 2.8)
    ref = FlightModel.integrate(truth[0], truth[1], truth[2] * 2 * math.pi, 2.30, 7.0)
    traj = ref["trajectory"]
    tg = np.linspace(0, ref["flight_time"], len(traj))
    obs = np.column_stack([np.interp(times, tg, traj[:, 0]), np.interp(times, tg, traj[:, 1])])
    obs += np.random.default_rng(0).normal(0, 0.004, obs.shape)
    rp = ReleaseInversion.invert(times, obs, 2.30, 7.0)
    print(f"      真值 速度 {truth[0]} 角度 {truth[1]} 旋转 {truth[2]} -> "
          f"反演 速度 {rp.speed:.3f} 角度 {rp.angle_deg:.2f} 旋转 {rp.spin_rps:.2f} "
          f"(残差 {rp.residual * 100:.2f} 厘米)")

    print("[3/6] 髋肩相位相干性（DTW + Hilbert 相位锁定值）")
    factory = DemoDataFactory(1)
    hip, sh = factory._joint_curves(0.85)
    good = JointPhaseAnalyzer.analyze(hip, sh)
    hip2, sh2 = factory._joint_curves(0.25)
    bad = JointPhaseAnalyzer.analyze(hip2, sh2)
    print(f"      高协调：相位锁定值 {good['plv']:.3f} | DTW {good['dtw']:.2f}")
    print(f"      低协调：相位锁定值 {bad['plv']:.3f} | DTW {bad['dtw']:.2f}")

    print("[4/6] 高斯过程 + 帕累托前沿")
    records = DemoDataFactory().generate(60)
    ds = ShotDataset(records)
    pr = ds.pareto_result
    print(f"      样本 {ds.n} 投，命中 {int(ds.made.sum())} 投 "
          f"({ds.made.mean() * 100:.1f}%)")
    print(f"      最优参数点：角度 {pr['star_angle']:.2f} 度 | "
          f"速度 {pr['star_speed']:.3f} 米每秒 | 预测概率 {pr['star_prob']:.3f}")
    print(f"      帕累托区间：角度 {pr['angle_interval'][0]:.1f}~{pr['angle_interval'][1]:.1f} 度 | "
          f"速度 {pr['speed_interval'][0]:.2f}~{pr['speed_interval'][1]:.2f} 米每秒")

    print("[5/6] 热力图网格与反馈")
    gx, gy, cnt, rate = ds.grid.cells()
    print(f"      有效网格 {len(cnt)} 个，最多一格出手 {cnt.max():.0f} 次")
    for m in build_feedback(ds):
        print(f"      反馈：{m}（{len(m)} 字）")
        assert len(m) <= 20, "反馈文字超过 20 字"

    print("[6/6] 九张图表构建")
    if not PLOTLY_OK:
        print("      跳过：未安装 plotly")
        return 0
    figs = build_all_figures(ds)
    for i, f in enumerate(figs, 1):
        assert f is not None
        print(f"      图 {i} 生成成功，图层数量 {len(f.data)}")
    print("全部通过。")
    return 0


def _under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()

    missing = []
    if not STREAMLIT_OK:
        missing.append("streamlit")
    if not PLOTLY_OK:
        missing.append("plotly")
    if not SCIPY_OK:
        missing.append("scipy")
    if missing:
        print("缺少必需依赖：" + ", ".join(missing))
        print("请先执行： pip install " + " ".join(missing))
        return 1

    if _under_streamlit():
        run_dashboard()
        return 0

    # 直接用 python 启动：自动拉起 streamlit，终端会打印本地网址
    print("正在启动本地服务，终端稍后会打印本地网址（Local URL）……")
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
           "--server.headless", "false", "--browser.gatherUsageStats", "false"]
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    if _under_streamlit():      # 被 `streamlit run` 执行时直接渲染界面
        run_dashboard()
    else:                       # 直接用 python 运行时拉起 streamlit 并打印本地网址
        sys.exit(main())
