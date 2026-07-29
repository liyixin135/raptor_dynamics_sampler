#!/usr/bin/env python3
"""Generate RAPTOR L2F-style quadrotor dynamics parameter JSON files.

This script is a standalone Python implementation aligned with RAPTOR's
foundation_policy/pre_training/sample_dynamics_parameters.cpp and the
rl_tools l2f sample_initial_parameters() logic.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List


N_ROTORS = 4


def nominal_parameters() -> Dict[str, Any]:
    """Return Crazyflie defaults from RAPTOR's l2f dynamics registry."""
    dynamics = {
        # Rotor positions in body frame, meters.
        "rotor_positions": [
            [0.028, -0.028, 0.0],
            [-0.028, -0.028, 0.0],
            [-0.028, 0.028, 0.0],
            [0.028, 0.028, 0.0],
        ],
        # Unit vectors for positive thrust direction of each rotor.
        "rotor_thrust_directions": [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        # Unit vectors for rotor drag torque direction.
        "rotor_torque_directions": [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        # Quadratic thrust curve coefficients: thrust = c0 + c1*u + c2*u^2, newtons.
        "rotor_thrust_coefficients": [
            [0.00352526, 0.01437313, 0.09223048],
            [0.00352526, 0.01437313, 0.09223048],
            [0.00352526, 0.01437313, 0.09223048],
            [0.00352526, 0.01437313, 0.09223048],
        ],
        # Rotor drag torque coefficient, meters in RAPTOR's thrust-to-torque model.
        "rotor_torque_constants": [4.665e-3] * N_ROTORS,
        # First-order motor response time constants while commanded thrust is rising, seconds.
        "rotor_time_constants_rising": [0.05545454545454546] * N_ROTORS,
        # First-order motor response time constants while commanded thrust is falling, seconds.
        "rotor_time_constants_falling": [0.24939393939393945] * N_ROTORS,
        # Vehicle mass, kilograms.
        "mass": 0.027 + 0.0017 + 0.0003 + 0.0016,
        # Gravity vector in world frame, m/s^2.
        "gravity": [0.0, 0.0, -9.81],
        # Body inertia matrix, kg*m^2.
        "J": [
            [9.416556729130406e-06, 0.0, 0.0],
            [0.0, 9.644051701582312e-06, 0.0],
            [0.0, 0.0, 1.745951732253285e-05],
        ],
        # Inverse inertia matrix.
        "J_inv": [
            [106195.93007988465, 0.0, 0.0],
            [0.0, 103690.85846314249, 0.0],
            [0.0, 0.0, 57275.35197719487],
        ],
        # Hover throttle as fraction of action limits, dimensionless [0, 1].
        "hovering_throttle_relative": 0.7261389721508553,
        # Normalized motor command limits used by RAPTOR for this model.
        "action_limit": {"min": 0.0, "max": 1.0},
    }

    return {
        "dynamics": dynamics,
        "integration": {"dt": 0.01},
        "mdp": {
            "init": {
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
                "position": 0.0,
                "orientation": 0.0,
                "linear_velocity": 0.0,
                "angular_velocity": 0.0,
                "imu_acceleration": 0.0,
            },
            "action_noise": {"normalized_rpm": 0.0},
            "termination": {
                "enabled": True,
                "position_threshold": 1.0,
                "linear_velocity_threshold": 2.0,
                "angular_velocity_threshold": 35.0,
                "position_integral_threshold": 10000.0,
                "orientation_integral_threshold": 50000.0,
            },
        },
        "disturbances": {
            "random_force": {"mean": 0.0, "std": 0.0},
            "random_torque": {"mean": 0.0, "std": 0.0},
        },
        "domain_randomization": domain_randomization_disabled(),
        "trajectory": {
            "MIXTURE_N": 2,
            "mixture": [0.5, 0.5],
            "langevin": {"gamma": 1.0, "omega": 2.0, "sigma": 0.5, "alpha": 0.01},
        },
    }


def raptor_sampling_ranges() -> Dict[str, float]:
    """Domain-randomization ranges from RAPTOR's pre-training sampler."""
    return {
        "thrust_to_weight_min": 1.5,
        "thrust_to_weight_max": 5.0,
        "torque_to_inertia_min": 40.0,
        "torque_to_inertia_max": 1200.0,
        "mass_min": 0.02,
        "mass_max": 5.0,
        "mass_size_deviation": 0.1,
        "rotor_time_constant_rising_min": 0.03,
        "rotor_time_constant_rising_max": 0.10,
        "rotor_time_constant_falling_min": 0.03,
        "rotor_time_constant_falling_max": 0.30,
        "rotor_torque_constant_min": 0.005,
        "rotor_torque_constant_max": 0.05,
        "orientation_offset_angle_max": 0.0,
        "disturbance_force_max": 0.3,
    }


def domain_randomization_disabled() -> Dict[str, float]:
    """RAPTOR writes sampled files with randomization disabled after sampling."""
    return {key: 0.0 for key in raptor_sampling_ranges()}


def vector_norm(values: List[float]) -> float:
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
    max_action = dynamics["action_limit"]["max"]
    total = 0.0
    for c0, c1, c2 in dynamics["rotor_thrust_coefficients"]:
        total += c0 + c1 * max_action + c2 * max_action * max_action
    return total


def sample_parameters(rng: random.Random) -> Dict[str, Any]:
    """Sample one parameter set with RAPTOR's core ordering."""
    params = nominal_parameters()
    ranges = raptor_sampling_ranges()
    dynamics = params["dynamics"]

    gravity_norm = vector_norm(dynamics["gravity"])
    thrust_to_weight_nominal = max_total_thrust(dynamics) / (dynamics["mass"] * gravity_norm)

    # 1. Sample target maximum thrust-to-weight ratio, dimensionless.
    thrust_to_weight = rng.uniform(
        ranges["thrust_to_weight_min"], ranges["thrust_to_weight_max"]
    )
    factor_thrust_to_weight = thrust_to_weight / thrust_to_weight_nominal

    # 2. Sample mass uniformly in linear size; mass scales with size^3.
    mass_nominal = dynamics["mass"]
    relative_size_min = cbrt(ranges["mass_min"])
    relative_size_max = cbrt(ranges["mass_max"])
    size_new = rng.uniform(relative_size_min, relative_size_max)
    mass_new = min(max(size_new**3, ranges["mass_min"]), ranges["mass_max"])
    scale_relative = cbrt(mass_new / mass_nominal)
    factor_mass = mass_new / mass_nominal
    dynamics["mass"] = mass_new

    # 3. Scale thrust curves so max thrust matches sampled mass and thrust-to-weight.
    factor_thrust_coefficients = factor_thrust_to_weight * factor_mass
    for rotor_coeffs in dynamics["rotor_thrust_coefficients"]:
        for order_i in range(3):
            rotor_coeffs[order_i] *= factor_thrust_coefficients

    # 4. Sample torque-to-inertia ratio and adjust diagonal inertia consistently.
    max_thrust_per_rotor = thrust_to_weight * dynamics["mass"] * gravity_norm / N_ROTORS
    first_rotor_distance_nominal = abs(dynamics["rotor_positions"][0][0])
    max_torque = first_rotor_distance_nominal * math.sqrt(2.0) * max_thrust_per_rotor
    torque_to_inertia_nominal = max_torque / dynamics["J"][0][0]
    torque_to_inertia = rng.uniform(
        ranges["torque_to_inertia_min"], ranges["torque_to_inertia_max"]
    )
    torque_to_inertia_factor = torque_to_inertia / torque_to_inertia_nominal

    # 5. Scale arm length from mass-derived size with an additional random deviation.
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

    # 6. Use one sampled yaw torque constant for all four rotors.
    torque_constant = rng.uniform(
        ranges["rotor_torque_constant_min"], ranges["rotor_torque_constant_max"]
    )
    dynamics["rotor_torque_constants"] = [torque_constant] * N_ROTORS

    # 7. Match RAPTOR's disturbance-force randomization for sampled files.
    surplus_thrust_to_weight = max(0.0, thrust_to_weight - 1.0)
    disturbance_multiple = rng.uniform(
        0.0, surplus_thrust_to_weight * ranges["disturbance_force_max"]
    )
    disturbance_force_std = disturbance_multiple * thrust_to_weight * dynamics["mass"] / 3.0
    params["disturbances"]["random_force"] = {"mean": 0.0, "std": disturbance_force_std}

    # 8. Sample first-order motor response time constants, seconds.
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

    # RAPTOR does not recompute hovering_throttle_relative before saving sampled JSON.
    # It disables future domain randomization in saved pre-training JSON files.
    params["domain_randomization"] = domain_randomization_disabled()
    return params


def write_samples(num: int, seed: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for index in range(num):
        params = sample_parameters(rng)
        output_path = output_dir / f"{index}.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(params, output_file, indent=2, sort_keys=False)
            output_file.write("\n")


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    if args.num < 0:
        raise ValueError("--num must be non-negative")
    write_samples(args.num, args.seed, args.output_dir)
    print(f"wrote {args.num} parameter files to {args.output_dir}")


if __name__ == "__main__":
    main()
