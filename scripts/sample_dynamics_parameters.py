#!/usr/bin/env python3
"""Generate RAPTOR-style quadrotor dynamics parameter JSON files.

This script is a standalone Python implementation inspired by RAPTOR's
foundation_policy/pre_training/sample_dynamics_parameters.cpp and the
rl_tools l2f sample_initial_parameters() logic.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List


# RAPTOR 这里使用的是四旋翼模型，因此所有 rotor 相关数组长度都是 4。
N_ROTORS = 4


def nominal_parameters() -> Dict[str, Any]:
    """Return Crazyflie-based nominal parameters used as the sampling baseline."""
    dynamics = {
        # 机体系下的旋翼位置，单位是米。
        "rotor_positions": [
            [0.028, -0.028, 0.0],
            [-0.028, -0.028, 0.0],
            [-0.028, 0.028, 0.0],
            [0.028, 0.028, 0.0],
        ],
        # 每个旋翼正推力方向的单位向量。
        "rotor_thrust_directions": [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        # 每个旋翼阻力力矩方向的单位向量。
        "rotor_torque_directions": [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        # 二次推力曲线系数：thrust = c0 + c1*u + c2*u^2，单位是牛顿。
        "rotor_thrust_coefficients": [
            [0.00352526, 0.01437313, 0.09223048],
            [0.00352526, 0.01437313, 0.09223048],
            [0.00352526, 0.01437313, 0.09223048],
            [0.00352526, 0.01437313, 0.09223048],
        ],
        # 旋翼阻力力矩系数；在 RAPTOR 的推力到力矩模型中量纲为米。
        "rotor_torque_constants": [4.665e-3] * N_ROTORS,
        # 指令推力上升时的一阶电机响应时间常数，单位是秒。
        "rotor_time_constants_rising": [0.05545454545454546] * N_ROTORS,
        # 指令推力下降时的一阶电机响应时间常数，单位是秒。
        "rotor_time_constants_falling": [0.24939393939393945] * N_ROTORS,
        # 飞行器质量，单位是千克。
        "mass": 0.027 + 0.0017 + 0.0003 + 0.0016,
        # 世界系下的重力向量，单位是 m/s^2。
        "gravity": [0.0, 0.0, -9.81],
        # 机体惯量矩阵，单位是 kg*m^2。
        "J": [
            [9.416556729130406e-06, 0.0, 0.0],
            [0.0, 9.644051701582312e-06, 0.0],
            [0.0, 0.0, 1.745951732253285e-05],
        ],
        # 惯量矩阵的逆矩阵。
        "J_inv": [
            [106195.93007988465, 0.0, 0.0],
            [0.0, 103690.85846314249, 0.0],
            [0.0, 0.0, 57275.35197719487],
        ],
        # 悬停油门在动作上下限中的相对比例，无量纲，范围 [0, 1]。
        "hovering_throttle_relative": 0.7261389721508553,
        # RAPTOR 在该模型中使用的归一化电机指令上下限。
        "action_limit": {"min": 0.0, "max": 1.0},
    }

    # 顶层结构尽量贴近 rl_tools 输出的 JSON，方便下游代码按熟悉路径读取字段，
    # 例如 parameters["dynamics"]["mass"]。
    return {
        "dynamics": dynamics,
        "integration": {"dt": 0.01},
        "mdp": {
            "init": {
                # 训练环境使用的初始状态范围。
                "guidance": 0.0,
                "max_position": 1.0,
                "max_angle": math.pi / 2.0,
                "max_linear_velocity": 0.0,
                "max_angular_velocity": 0.0,
                "relative_rpm": True,
                "min_rpm": -1.0,
                "max_rpm": 1.0,
            },
            "reward": {
                # 奖励权重对应 RAPTOR 采样器中的 overwrite() 配置。
                "non_negative": False,
                "scale": 1.0,
                "constant": 1.5,
                "termination_penalty": -100.0,
                "position": 1.0,
                "position_clip": 0.0,
                "orientation": 0.1,
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
                "linear_acceleration": 0.0,
                "angular_acceleration": 0.0,
                "action": 0.0,
                "d_action": 1.0,
                "position_error_integral": 0.0,
            },
            "observation_noise": {
                # 仿真传感器噪声的标准差。
                "position": 0.0,
                "orientation": 0.0,
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
                "imu_acceleration": 0.0,
            },
            "action_noise": {"normalized_rpm": 0.0},
            "termination": {
                # 回合终止阈值；位置相关阈值会在臂长采样后重新缩放。
                "enabled": True,
                "position_threshold": 1.0,
                "linear_velocity_threshold": 2.0,
                "angular_velocity_threshold": 35.0,
                "position_integral_threshold": 10000.0,
                "orientation_integral_threshold": 50000.0,
            },
        },
        "disturbances": {
            # 高斯外力/外力矩扰动；外力标准差会在后续采样。
            "random_force": {"mean": 0.0, "std": 0.0},
            "random_torque": {"mean": 0.0, "std": 0.0},
        },
        "domain_randomization": domain_randomization_disabled(),
        "trajectory": {
            # 保留 RAPTOR 的 trajectory 组件，保证 JSON 结构兼容。
            "MIXTURE_N": 2,
            "mixture": [0.5, 0.5],
            "langevin": {"gamma": 1.0, "omega": 2.0, "sigma": 0.5, "alpha": 0.01},
        },
    }


def raptor_sampling_ranges() -> Dict[str, float]:
    """Domain-randomization ranges copied from RAPTOR's pre-training sampler."""
    return {
        # 最大可用推力除以机体重量，无量纲。
        "thrust_to_weight_min": 1.5,
        "thrust_to_weight_max": 5.0,
        # 近似最大机体力矩除以横滚惯量，单位是 rad/s^2。
        "torque_to_inertia_min": 40.0,
        "torque_to_inertia_max": 1200.0,
        # 质量范围，单位是千克；下面会在质量立方根上做均匀采样。
        "mass_min": 0.02,
        "mass_max": 5.0,
        # 在由质量推导出的臂长附近增加额外倒数尺度扰动。
        "mass_size_deviation": 0.1,
        # 电机一阶响应时间常数范围，单位是秒。
        "rotor_time_constant_rising_min": 0.03,
        "rotor_time_constant_rising_max": 0.10,
        "rotor_time_constant_falling_min": 0.03,
        "rotor_time_constant_falling_max": 0.30,
        # 推力到阻力力矩模型中使用的旋翼偏航力矩系数范围。
        "rotor_torque_constant_min": 0.005,
        "rotor_torque_constant_max": 0.05,
        "orientation_offset_angle_max": 0.0,
        "disturbance_force_max": 0.3,
    }


def domain_randomization_disabled() -> Dict[str, float]:
    """RAPTOR writes sampled files with randomization disabled after sampling."""
    # 对齐 C++ 采样器在保存前对 params_copy.domain_randomization 的赋值。
    return {key: 0.0 for key in raptor_sampling_ranges()}


def vector_norm(values: List[float]) -> float:
    """计算重力、旋翼位置等 3D 向量的欧氏范数。"""
    return math.sqrt(sum(value * value for value in values))


def cbrt(value: float) -> float:
    """Real-valued cube root compatible with Python versions without math.cbrt."""
    return math.copysign(abs(value) ** (1.0 / 3.0), value)


def sample_domain_randomization_factor(rng: random.Random, value_range: float) -> float:
    """Reciprocal scale factor used by RLtools for size deviations.

    RAPTOR calls normal_distribution::sample(device.random, -range, range, rng).
    We map that to Python as Gaussian(mean=-range, std=range). Positive values
    become 1 + x, while negative values become 1 / (1 - x).
    """
    factor = rng.gauss(-value_range, value_range)
    return 1.0 / (1.0 - factor) if factor < 0.0 else 1.0 + factor


def max_total_thrust(dynamics: Dict[str, Any]) -> float:
    """计算最大动作指令下四个旋翼的总推力。"""
    max_action = dynamics["action_limit"]["max"]
    total = 0.0
    for c0, c1, c2 in dynamics["rotor_thrust_coefficients"]:
        total += c0 + c1 * max_action + c2 * max_action * max_action
    return total


def update_hovering_throttle(dynamics: Dict[str, Any]) -> None:
    """根据平均二次推力曲线重新计算悬停油门比例。"""
    per_rotor_hover_thrust = dynamics["mass"] * vector_norm(dynamics["gravity"]) / N_ROTORS
    c0 = sum(coeffs[0] for coeffs in dynamics["rotor_thrust_coefficients"]) / N_ROTORS
    c1 = sum(coeffs[1] for coeffs in dynamics["rotor_thrust_coefficients"]) / N_ROTORS
    c2 = sum(coeffs[2] for coeffs in dynamics["rotor_thrust_coefficients"]) / N_ROTORS
    # 求解 c0 + c1*u + c2*u^2 = m*g/4，取正的悬停指令。
    if abs(c2) < 1e-12:
        throttle = (per_rotor_hover_thrust - c0) / c1
    else:
        discriminant = max(0.0, c1 * c1 - 4.0 * c2 * (c0 - per_rotor_hover_thrust))
        throttle = (-c1 + math.sqrt(discriminant)) / (2.0 * c2)
    min_action = dynamics["action_limit"]["min"]
    max_action = dynamics["action_limit"]["max"]
    dynamics["hovering_throttle_relative"] = (throttle - min_action) / (max_action - min_action)


def sample_parameters(rng: random.Random) -> Dict[str, Any]:
    """Sample one parameter set with the same core ordering as RAPTOR."""
    params = nominal_parameters()
    ranges = raptor_sampling_ranges()
    dynamics = params["dynamics"]

    gravity_norm = vector_norm(dynamics["gravity"])
    thrust_to_weight_nominal = max_total_thrust(dynamics) / (dynamics["mass"] * gravity_norm)

    # 1. 采样目标最大推重比，无量纲。
    thrust_to_weight = rng.uniform(
        ranges["thrust_to_weight_min"], ranges["thrust_to_weight_max"]
    )
    factor_thrust_to_weight = thrust_to_weight / thrust_to_weight_nominal

    # 2. 在线性尺寸上均匀采样质量；质量按尺寸的三次方缩放。
    mass_nominal = dynamics["mass"]
    relative_size_min = cbrt(ranges["mass_min"])
    relative_size_max = cbrt(ranges["mass_max"])
    size_new = rng.uniform(relative_size_min, relative_size_max)
    mass_new = min(max(size_new**3, ranges["mass_min"]), ranges["mass_max"])
    scale_relative = cbrt(mass_new / mass_nominal)
    factor_mass = mass_new / mass_nominal
    dynamics["mass"] = mass_new

    # 3. 缩放推力曲线，使最大推力匹配采样得到的质量和推重比。
    factor_thrust_coefficients = factor_thrust_to_weight * factor_mass
    for rotor_coeffs in dynamics["rotor_thrust_coefficients"]:
        for order_i in range(3):
            rotor_coeffs[order_i] *= factor_thrust_coefficients

    # 4. 采样力矩惯量比，并同步调整惯量矩阵的对角项。
    max_thrust_per_rotor = thrust_to_weight * dynamics["mass"] * gravity_norm / N_ROTORS
    first_rotor_distance_nominal = abs(dynamics["rotor_positions"][0][0])
    max_torque = first_rotor_distance_nominal * math.sqrt(2.0) * max_thrust_per_rotor
    torque_to_inertia_nominal = max_torque / dynamics["J"][0][0]
    torque_to_inertia = rng.uniform(
        ranges["torque_to_inertia_min"], ranges["torque_to_inertia_max"]
    )
    torque_to_inertia_factor = torque_to_inertia / torque_to_inertia_nominal

    # 5. 根据质量推导出的尺寸缩放臂长，并叠加额外随机偏差。
    size_factor = sample_domain_randomization_factor(rng, ranges["mass_size_deviation"])
    rotor_distance_factor = scale_relative * size_factor
    inertia_factor = torque_to_inertia_factor / rotor_distance_factor
    for axis_i in range(3):
        dynamics["J"][axis_i][axis_i] /= inertia_factor
        dynamics["J_inv"][axis_i][axis_i] *= inertia_factor
    for rotor_position in dynamics["rotor_positions"]:
        for axis_i in range(3):
            rotor_position[axis_i] *= rotor_distance_factor

    max_rotor_distance = max(vector_norm(position) for position in dynamics["rotor_positions"])
    params["mdp"]["termination"]["position_threshold"] = max_rotor_distance * 20.0
    params["mdp"]["init"]["max_position"] = max_rotor_distance * 10.0

    # 6. 四个旋翼共用同一个采样得到的偏航力矩常数。
    torque_constant = rng.uniform(
        ranges["rotor_torque_constant_min"], ranges["rotor_torque_constant_max"]
    )
    dynamics["rotor_torque_constants"] = [torque_constant] * N_ROTORS

    # 7. 对齐 RAPTOR 保存采样文件时使用的随机外力扰动。
    surplus_thrust_to_weight = max(0.0, thrust_to_weight - 1.0)
    disturbance_multiple = rng.uniform(
        0.0, surplus_thrust_to_weight * ranges["disturbance_force_max"]
    )
    disturbance_force_std = disturbance_multiple * thrust_to_weight * dynamics["mass"] / 3.0
    params["disturbances"]["random_force"] = {"mean": 0.0, "std": disturbance_force_std}

    # 8. 采样一阶电机响应时间常数，单位是秒。
    rising = rng.uniform(
        ranges["rotor_time_constant_rising_min"],
        ranges["rotor_time_constant_rising_max"],
    )
    falling = rng.uniform(
        ranges["rotor_time_constant_falling_min"],
        ranges["rotor_time_constant_falling_max"],
    )
    dynamics["rotor_time_constants_rising"] = [rising] * N_ROTORS
    dynamics["rotor_time_constants_falling"] = [falling] * N_ROTORS

    # RAPTOR 在保存预训练 JSON 文件后会禁用后续 domain randomization。
    params["domain_randomization"] = domain_randomization_disabled()
    return params


def write_samples(num: int, seed: int, output_dir: Path) -> None:
    """使用确定性随机种子写出按编号命名的 JSON 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index in range(num):
        params = sample_parameters(rng)
        output_path = output_dir / f"{index}.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(params, output_file, indent=2, sort_keys=False)
            output_file.write("\n")


def parse_args() -> argparse.Namespace:
    """解析批量生成实验所需的简单命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Generate RAPTOR-style quadrotor dynamics parameter JSON files."
    )
    parser.add_argument("--num", type=int, default=1000, help="number of JSON files to generate")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dynamics_parameters"),
        help="directory where 0.json, 1.json, ... are written",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    if args.num < 0:
        raise ValueError("--num must be non-negative")
    write_samples(args.num, args.seed, args.output_dir)
    print(f"wrote {args.num} parameter files to {args.output_dir}")


if __name__ == "__main__":
    main()
