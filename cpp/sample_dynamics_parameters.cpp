#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// 这个程序是 scripts/sample_dynamics_parameters.py 的 C++17 版本。
// 目标只做一件事：随机生成四旋翼动力学参数 JSON 文件。
// 它不依赖 RAPTOR/rl_tools 头文件，方便初学者直接用 g++ 编译运行。

namespace fs = std::filesystem;

constexpr int N_ROTORS = 4;
constexpr double PI = 3.14159265358979323846;

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;
using RotorCoefficients = std::array<double, 3>;

struct ActionLimit {
    double min = 0.0;
    double max = 1.0;
};

struct Dynamics {
    std::array<Vec3, N_ROTORS> rotor_positions{};
    std::array<Vec3, N_ROTORS> rotor_thrust_directions{};
    std::array<Vec3, N_ROTORS> rotor_torque_directions{};
    std::array<RotorCoefficients, N_ROTORS> rotor_thrust_coefficients{};
    std::array<double, N_ROTORS> rotor_torque_constants{};
    std::array<double, N_ROTORS> rotor_time_constants_rising{};
    std::array<double, N_ROTORS> rotor_time_constants_falling{};
    double mass = 0.0;
    Vec3 gravity{};
    Mat3 J{};
    Mat3 J_inv{};
    double hovering_throttle_relative = 0.0;
    ActionLimit action_limit{};
};

struct RangeConfig {
    double thrust_to_weight_min = 1.5;
    double thrust_to_weight_max = 5.0;
    double torque_to_inertia_min = 40.0;
    double torque_to_inertia_max = 1200.0;
    double mass_min = 0.02;
    double mass_max = 5.0;
    double mass_size_deviation = 0.1;
    double rotor_time_constant_rising_min = 0.03;
    double rotor_time_constant_rising_max = 0.10;
    double rotor_time_constant_falling_min = 0.03;
    double rotor_time_constant_falling_max = 0.30;
    double rotor_torque_constant_min = 0.005;
    double rotor_torque_constant_max = 0.05;
    double orientation_offset_angle_max = 0.0;
    double disturbance_force_max = 0.3;
};

struct InitConfig {
    double guidance = 0.0;
    double max_position = 1.0;
    double max_angle = PI / 2.0;
    double max_linear_velocity = 0.0;
    double max_angular_velocity = 0.0;
    bool relative_rpm = true;
    double min_rpm = -1.0;
    double max_rpm = 1.0;
};

struct TerminationConfig {
    bool enabled = true;
    double position_threshold = 1.0;
    double linear_velocity_threshold = 2.0;
    double angular_velocity_threshold = 35.0;
    double position_integral_threshold = 10000.0;
    double orientation_integral_threshold = 50000.0;
};

struct Disturbance {
    double mean = 0.0;
    double std = 0.0;
};

struct Parameters {
    Dynamics dynamics{};
    InitConfig init{};
    TerminationConfig termination{};
    Disturbance random_force{};
    Disturbance random_torque{};
    RangeConfig domain_randomization{};
};

struct ProgramOptions {
    int num = 1000;
    unsigned int seed = 0;
    fs::path output_dir = "dynamics_parameters";
};

double vector_norm(const Vec3& values) {
    return std::sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

double uniform(std::mt19937& rng, double min_value, double max_value) {
    std::uniform_real_distribution<double> distribution(min_value, max_value);
    return distribution(rng);
}

double gaussian(std::mt19937& rng, double mean, double stddev) {
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(rng);
}

void fill_all(std::array<double, N_ROTORS>& values, double value) {
    for (double& item : values) {
        item = value;
    }
}

RangeConfig raptor_sampling_ranges() {
    // 这些范围来自 RAPTOR 预训练采样器，也是 Python 版本使用的范围。
    return RangeConfig{};
}

RangeConfig domain_randomization_disabled() {
    // 保存到 JSON 的是已经采样好的确定参数，因此关闭后续 domain randomization。
    RangeConfig disabled{};
    disabled.thrust_to_weight_min = 0.0;
    disabled.thrust_to_weight_max = 0.0;
    disabled.torque_to_inertia_min = 0.0;
    disabled.torque_to_inertia_max = 0.0;
    disabled.mass_min = 0.0;
    disabled.mass_max = 0.0;
    disabled.mass_size_deviation = 0.0;
    disabled.rotor_time_constant_rising_min = 0.0;
    disabled.rotor_time_constant_rising_max = 0.0;
    disabled.rotor_time_constant_falling_min = 0.0;
    disabled.rotor_time_constant_falling_max = 0.0;
    disabled.rotor_torque_constant_min = 0.0;
    disabled.rotor_torque_constant_max = 0.0;
    disabled.orientation_offset_angle_max = 0.0;
    disabled.disturbance_force_max = 0.0;
    return disabled;
}

Parameters nominal_parameters() {
    // 基础模板使用 Crazyflie 风格参数，和 Python 版本保持一致。
    Parameters params;
    Dynamics& d = params.dynamics;

    d.rotor_positions = {{{{0.028, -0.028, 0.0}},
                          {{-0.028, -0.028, 0.0}},
                          {{-0.028, 0.028, 0.0}},
                          {{0.028, 0.028, 0.0}}}};
    d.rotor_thrust_directions = {{{{0.0, 0.0, 1.0}},
                                  {{0.0, 0.0, 1.0}},
                                  {{0.0, 0.0, 1.0}},
                                  {{0.0, 0.0, 1.0}}}};
    d.rotor_torque_directions = {{{{0.0, 0.0, -1.0}},
                                  {{0.0, 0.0, 1.0}},
                                  {{0.0, 0.0, -1.0}},
                                  {{0.0, 0.0, 1.0}}}};
    d.rotor_thrust_coefficients = {{{{0.00352526, 0.01437313, 0.09223048}},
                                    {{0.00352526, 0.01437313, 0.09223048}},
                                    {{0.00352526, 0.01437313, 0.09223048}},
                                    {{0.00352526, 0.01437313, 0.09223048}}}};
    fill_all(d.rotor_torque_constants, 4.665e-3);
    fill_all(d.rotor_time_constants_rising, 0.05545454545454546);
    fill_all(d.rotor_time_constants_falling, 0.24939393939393945);
    d.mass = 0.027 + 0.0017 + 0.0003 + 0.0016;
    d.gravity = {{0.0, 0.0, -9.81}};
    d.J = {{{{9.416556729130406e-06, 0.0, 0.0}},
            {{0.0, 9.644051701582312e-06, 0.0}},
            {{0.0, 0.0, 1.745951732253285e-05}}}};
    d.J_inv = {{{{106195.93007988465, 0.0, 0.0}},
                {{0.0, 103690.85846314249, 0.0}},
                {{0.0, 0.0, 57275.35197719487}}}};
    d.hovering_throttle_relative = 0.7261389721508553;
    d.action_limit = {0.0, 1.0};

    params.domain_randomization = domain_randomization_disabled();
    return params;
}

double max_total_thrust(const Dynamics& dynamics) {
    // 每个电机最大推力：thrust = c0 + c1*u + c2*u^2，这里 u 取 action_limit.max。
    const double max_action = dynamics.action_limit.max;
    double total = 0.0;
    for (const auto& coeffs : dynamics.rotor_thrust_coefficients) {
        total += coeffs[0] + coeffs[1] * max_action + coeffs[2] * max_action * max_action;
    }
    return total;
}

void update_hovering_throttle(Dynamics& dynamics) {
    // 悬停时四个电机总推力等于重力：sum(thrust_i) = mass * |gravity|。
    const double per_rotor_hover_thrust = dynamics.mass * vector_norm(dynamics.gravity) / N_ROTORS;
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    for (const auto& coeffs : dynamics.rotor_thrust_coefficients) {
        c0 += coeffs[0];
        c1 += coeffs[1];
        c2 += coeffs[2];
    }
    c0 /= N_ROTORS;
    c1 /= N_ROTORS;
    c2 /= N_ROTORS;

    double throttle = 0.0;
    if (std::abs(c2) < 1e-12) {
        // 如果二次项非常小，就退化为线性方程：c0 + c1*u = hover_thrust。
        throttle = (per_rotor_hover_thrust - c0) / c1;
    } else {
        // 解二次方程：c2*u^2 + c1*u + (c0 - hover_thrust) = 0。
        const double discriminant = std::max(0.0, c1 * c1 - 4.0 * c2 * (c0 - per_rotor_hover_thrust));
        throttle = (-c1 + std::sqrt(discriminant)) / (2.0 * c2);
    }

    const double min_action = dynamics.action_limit.min;
    const double max_action = dynamics.action_limit.max;
    dynamics.hovering_throttle_relative = (throttle - min_action) / (max_action - min_action);
}

double sample_domain_randomization_factor(std::mt19937& rng, double value_range) {
    // RAPTOR/rl_tools 对尺寸偏差使用一个 reciprocal scale factor。
    // x >= 0 时放大为 1 + x；x < 0 时缩小为 1 / (1 - x)。
    const double x = gaussian(rng, -value_range, value_range);
    if (x < 0.0) {
        return 1.0 / (1.0 - x);
    }
    return 1.0 + x;
}

Parameters sample_parameters(std::mt19937& rng) {
    Parameters params = nominal_parameters();
    const RangeConfig ranges = raptor_sampling_ranges();
    Dynamics& d = params.dynamics;

    const double gravity_norm = vector_norm(d.gravity);
    const double mass_nominal = d.mass;
    const double thrust_to_weight_nominal = max_total_thrust(d) / (mass_nominal * gravity_norm);

    // 1. 随机采样质量。为了覆盖大范围质量，先均匀采样“尺寸”，再用 mass = size^3。
    const double relative_size_min = std::cbrt(ranges.mass_min);
    const double relative_size_max = std::cbrt(ranges.mass_max);
    const double size_new = uniform(rng, relative_size_min, relative_size_max);
    const double mass_new = std::min(std::max(size_new * size_new * size_new, ranges.mass_min), ranges.mass_max);
    const double scale_relative = std::cbrt(mass_new / mass_nominal);
    const double factor_mass = mass_new / mass_nominal;
    d.mass = mass_new;

    // 2. 随机采样最大推重比 thrust_to_weight = max_total_thrust / (mass * g)。
    const double thrust_to_weight = uniform(rng, ranges.thrust_to_weight_min, ranges.thrust_to_weight_max);
    const double factor_thrust_to_weight = thrust_to_weight / thrust_to_weight_nominal;

    // 3. 同时考虑新质量和目标推重比，缩放所有电机推力曲线系数。
    const double factor_thrust_coefficients = factor_thrust_to_weight * factor_mass;
    for (auto& rotor_coeffs : d.rotor_thrust_coefficients) {
        for (double& coeff : rotor_coeffs) {
            coeff *= factor_thrust_coefficients;
        }
    }

    // 4. 随机采样 torque_to_inertia，并通过调整惯量矩阵 J 来匹配目标角加速度能力。
    const double max_thrust_per_rotor = thrust_to_weight * d.mass * gravity_norm / N_ROTORS;
    const double first_rotor_distance_nominal = std::abs(d.rotor_positions[0][0]);
    const double max_torque = first_rotor_distance_nominal * std::sqrt(2.0) * max_thrust_per_rotor;
    const double torque_to_inertia_nominal = max_torque / d.J[0][0];
    const double torque_to_inertia = uniform(rng, ranges.torque_to_inertia_min, ranges.torque_to_inertia_max);
    const double torque_to_inertia_factor = torque_to_inertia / torque_to_inertia_nominal;

    // 5. 机臂长度随质量对应的尺寸变化，再叠加 mass_size_deviation 随机偏差。
    const double size_factor = sample_domain_randomization_factor(rng, ranges.mass_size_deviation);
    const double rotor_distance_factor = scale_relative * size_factor;
    const double inertia_factor = torque_to_inertia_factor / rotor_distance_factor;
    for (int axis = 0; axis < 3; ++axis) {
        // J 越大，同样力矩产生的角加速度越小；J_inv 是 J 的逆，所以反向缩放。
        d.J[axis][axis] /= inertia_factor;
        d.J_inv[axis][axis] *= inertia_factor;
    }
    for (auto& rotor_position : d.rotor_positions) {
        for (double& coordinate : rotor_position) {
            coordinate *= rotor_distance_factor;
        }
    }

    // 6. 尺寸变化后，终止阈值和初始位置范围也按最大电机距离更新。
    double max_rotor_distance = 0.0;
    for (const auto& rotor_position : d.rotor_positions) {
        max_rotor_distance = std::max(max_rotor_distance, vector_norm(rotor_position));
    }
    params.termination.position_threshold = max_rotor_distance * 20.0;
    params.init.max_position = max_rotor_distance * 10.0;

    // 7. 随机采样 rotor_torque_constant，四个电机使用同一个数值。
    const double torque_constant = uniform(rng, ranges.rotor_torque_constant_min, ranges.rotor_torque_constant_max);
    fill_all(d.rotor_torque_constants, torque_constant);

    // 8. 随机采样扰动力标准差。推重比越高，可承受扰动力范围越大。
    const double surplus_thrust_to_weight = std::max(0.0, thrust_to_weight - 1.0);
    const double disturbance_multiple = uniform(rng, 0.0, surplus_thrust_to_weight * ranges.disturbance_force_max);
    const double disturbance_force_std = disturbance_multiple * thrust_to_weight * d.mass / 3.0;
    params.random_force = {0.0, disturbance_force_std};

    // 9. 随机采样电机一阶响应时间常数：上升和下降分别采样。
    const double rising = uniform(rng, ranges.rotor_time_constant_rising_min, ranges.rotor_time_constant_rising_max);
    const double falling = uniform(rng, ranges.rotor_time_constant_falling_min, ranges.rotor_time_constant_falling_max);
    fill_all(d.rotor_time_constants_rising, rising);
    fill_all(d.rotor_time_constants_falling, falling);

    update_hovering_throttle(d);

    // 10. 保存前关闭 domain_randomization，避免读取 JSON 后再次随机化。
    params.domain_randomization = domain_randomization_disabled();
    return params;
}

std::string indent(int spaces) {
    return std::string(static_cast<std::size_t>(spaces), ' ');
}

void write_double(std::ostream& os, double value) {
    os << std::setprecision(17) << value;
}

void write_vec3(std::ostream& os, const Vec3& values) {
    os << "[";
    for (int i = 0; i < 3; ++i) {
        if (i > 0) {
            os << ", ";
        }
        write_double(os, values[i]);
    }
    os << "]";
}

void write_double_array(std::ostream& os, const std::array<double, N_ROTORS>& values) {
    os << "[";
    for (int i = 0; i < N_ROTORS; ++i) {
        if (i > 0) {
            os << ", ";
        }
        write_double(os, values[i]);
    }
    os << "]";
}

void write_vec3_array(std::ostream& os, const std::array<Vec3, N_ROTORS>& values, int spaces) {
    os << "[\n";
    for (int i = 0; i < N_ROTORS; ++i) {
        os << indent(spaces + 2);
        write_vec3(os, values[i]);
        os << (i + 1 == N_ROTORS ? "\n" : ",\n");
    }
    os << indent(spaces) << "]";
}

void write_coefficients_array(std::ostream& os, const std::array<RotorCoefficients, N_ROTORS>& values, int spaces) {
    os << "[\n";
    for (int i = 0; i < N_ROTORS; ++i) {
        os << indent(spaces + 2) << "[";
        for (int j = 0; j < 3; ++j) {
            if (j > 0) {
                os << ", ";
            }
            write_double(os, values[i][j]);
        }
        os << "]" << (i + 1 == N_ROTORS ? "\n" : ",\n");
    }
    os << indent(spaces) << "]";
}

void write_mat3(std::ostream& os, const Mat3& matrix, int spaces) {
    os << "[\n";
    for (int i = 0; i < 3; ++i) {
        os << indent(spaces + 2) << "[";
        for (int j = 0; j < 3; ++j) {
            if (j > 0) {
                os << ", ";
            }
            write_double(os, matrix[i][j]);
        }
        os << "]" << (i + 1 == 3 ? "\n" : ",\n");
    }
    os << indent(spaces) << "]";
}

void write_range_config(std::ostream& os, const RangeConfig& r, int spaces) {
    os << "{\n";
    os << indent(spaces + 2) << "\"thrust_to_weight_min\": "; write_double(os, r.thrust_to_weight_min); os << ",\n";
    os << indent(spaces + 2) << "\"thrust_to_weight_max\": "; write_double(os, r.thrust_to_weight_max); os << ",\n";
    os << indent(spaces + 2) << "\"torque_to_inertia_min\": "; write_double(os, r.torque_to_inertia_min); os << ",\n";
    os << indent(spaces + 2) << "\"torque_to_inertia_max\": "; write_double(os, r.torque_to_inertia_max); os << ",\n";
    os << indent(spaces + 2) << "\"mass_min\": "; write_double(os, r.mass_min); os << ",\n";
    os << indent(spaces + 2) << "\"mass_max\": "; write_double(os, r.mass_max); os << ",\n";
    os << indent(spaces + 2) << "\"mass_size_deviation\": "; write_double(os, r.mass_size_deviation); os << ",\n";
    os << indent(spaces + 2) << "\"rotor_time_constant_rising_min\": "; write_double(os, r.rotor_time_constant_rising_min); os << ",\n";
    os << indent(spaces + 2) << "\"rotor_time_constant_rising_max\": "; write_double(os, r.rotor_time_constant_rising_max); os << ",\n";
    os << indent(spaces + 2) << "\"rotor_time_constant_falling_min\": "; write_double(os, r.rotor_time_constant_falling_min); os << ",\n";
    os << indent(spaces + 2) << "\"rotor_time_constant_falling_max\": "; write_double(os, r.rotor_time_constant_falling_max); os << ",\n";
    os << indent(spaces + 2) << "\"rotor_torque_constant_min\": "; write_double(os, r.rotor_torque_constant_min); os << ",\n";
    os << indent(spaces + 2) << "\"rotor_torque_constant_max\": "; write_double(os, r.rotor_torque_constant_max); os << ",\n";
    os << indent(spaces + 2) << "\"orientation_offset_angle_max\": "; write_double(os, r.orientation_offset_angle_max); os << ",\n";
    os << indent(spaces + 2) << "\"disturbance_force_max\": "; write_double(os, r.disturbance_force_max); os << "\n";
    os << indent(spaces) << "}";
}

void write_json(std::ostream& os, const Parameters& p) {
    const Dynamics& d = p.dynamics;
    os << "{\n";
    os << "  \"dynamics\": {\n";
    os << "    \"rotor_positions\": "; write_vec3_array(os, d.rotor_positions, 4); os << ",\n";
    os << "    \"rotor_thrust_directions\": "; write_vec3_array(os, d.rotor_thrust_directions, 4); os << ",\n";
    os << "    \"rotor_torque_directions\": "; write_vec3_array(os, d.rotor_torque_directions, 4); os << ",\n";
    os << "    \"rotor_thrust_coefficients\": "; write_coefficients_array(os, d.rotor_thrust_coefficients, 4); os << ",\n";
    os << "    \"rotor_torque_constants\": "; write_double_array(os, d.rotor_torque_constants); os << ",\n";
    os << "    \"rotor_time_constants_rising\": "; write_double_array(os, d.rotor_time_constants_rising); os << ",\n";
    os << "    \"rotor_time_constants_falling\": "; write_double_array(os, d.rotor_time_constants_falling); os << ",\n";
    os << "    \"mass\": "; write_double(os, d.mass); os << ",\n";
    os << "    \"gravity\": "; write_vec3(os, d.gravity); os << ",\n";
    os << "    \"J\": "; write_mat3(os, d.J, 4); os << ",\n";
    os << "    \"J_inv\": "; write_mat3(os, d.J_inv, 4); os << ",\n";
    os << "    \"hovering_throttle_relative\": "; write_double(os, d.hovering_throttle_relative); os << ",\n";
    os << "    \"action_limit\": {\"min\": "; write_double(os, d.action_limit.min); os << ", \"max\": "; write_double(os, d.action_limit.max); os << "}\n";
    os << "  },\n";
    os << "  \"integration\": {\"dt\": 0.01},\n";
    os << "  \"mdp\": {\n";
    os << "    \"init\": {\n";
    os << "      \"guidance\": "; write_double(os, p.init.guidance); os << ",\n";
    os << "      \"max_position\": "; write_double(os, p.init.max_position); os << ",\n";
    os << "      \"max_angle\": "; write_double(os, p.init.max_angle); os << ",\n";
    os << "      \"max_linear_velocity\": "; write_double(os, p.init.max_linear_velocity); os << ",\n";
    os << "      \"max_angular_velocity\": "; write_double(os, p.init.max_angular_velocity); os << ",\n";
    os << "      \"relative_rpm\": true,\n";
    os << "      \"min_rpm\": "; write_double(os, p.init.min_rpm); os << ",\n";
    os << "      \"max_rpm\": "; write_double(os, p.init.max_rpm); os << "\n";
    os << "    },\n";
    os << "    \"reward\": {\n";
    os << "      \"non_negative\": false,\n";
    os << "      \"scale\": 1.0,\n";
    os << "      \"constant\": 1.5,\n";
    os << "      \"termination_penalty\": -100.0,\n";
    os << "      \"position\": 1.0,\n";
    os << "      \"position_clip\": 0.0,\n";
    os << "      \"orientation\": 0.1,\n";
    os << "      \"linear_velocity\": 0.0,\n";
    os << "      \"angular_velocity\": 0.0,\n";
    os << "      \"linear_acceleration\": 0.0,\n";
    os << "      \"angular_acceleration\": 0.0,\n";
    os << "      \"action\": 0.0,\n";
    os << "      \"d_action\": 1.0,\n";
    os << "      \"position_error_integral\": 0.0\n";
    os << "    },\n";
    os << "    \"observation_noise\": {\n";
    os << "      \"position\": 0.0,\n";
    os << "      \"orientation\": 0.0,\n";
    os << "      \"linear_velocity\": 0.0,\n";
    os << "      \"angular_velocity\": 0.0,\n";
    os << "      \"imu_acceleration\": 0.0\n";
    os << "    },\n";
    os << "    \"action_noise\": {\"normalized_rpm\": 0.0},\n";
    os << "    \"termination\": {\n";
    os << "      \"enabled\": true,\n";
    os << "      \"position_threshold\": "; write_double(os, p.termination.position_threshold); os << ",\n";
    os << "      \"linear_velocity_threshold\": "; write_double(os, p.termination.linear_velocity_threshold); os << ",\n";
    os << "      \"angular_velocity_threshold\": "; write_double(os, p.termination.angular_velocity_threshold); os << ",\n";
    os << "      \"position_integral_threshold\": "; write_double(os, p.termination.position_integral_threshold); os << ",\n";
    os << "      \"orientation_integral_threshold\": "; write_double(os, p.termination.orientation_integral_threshold); os << "\n";
    os << "    }\n";
    os << "  },\n";
    os << "  \"disturbances\": {\n";
    os << "    \"random_force\": {\"mean\": "; write_double(os, p.random_force.mean); os << ", \"std\": "; write_double(os, p.random_force.std); os << "},\n";
    os << "    \"random_torque\": {\"mean\": "; write_double(os, p.random_torque.mean); os << ", \"std\": "; write_double(os, p.random_torque.std); os << "}\n";
    os << "  },\n";
    os << "  \"domain_randomization\": "; write_range_config(os, p.domain_randomization, 2); os << ",\n";
    os << "  \"trajectory\": {\n";
    os << "    \"MIXTURE_N\": 2,\n";
    os << "    \"mixture\": [0.5, 0.5],\n";
    os << "    \"langevin\": {\"gamma\": 1.0, \"omega\": 2.0, \"sigma\": 0.5, \"alpha\": 0.01}\n";
    os << "  }\n";
    os << "}\n";
}

ProgramOptions parse_args(int argc, char** argv) {
    ProgramOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            ++i;
            return argv[i];
        };

        if (arg == "--num") {
            options.num = std::stoi(require_value(arg));
        } else if (arg == "--seed") {
            options.seed = static_cast<unsigned int>(std::stoul(require_value(arg)));
        } else if (arg == "--output-dir") {
            options.output_dir = require_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: sample_dynamics_parameters_cpp --num 10 --seed 0 --output-dir test_output_cpp\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.num < 0) {
        throw std::runtime_error("--num must be non-negative");
    }
    return options;
}

void write_samples(int num, unsigned int seed, const fs::path& output_dir) {
    fs::create_directories(output_dir);
    std::mt19937 rng(seed);

    for (int index = 0; index < num; ++index) {
        const Parameters params = sample_parameters(rng);
        const fs::path output_path = output_dir / (std::to_string(index) + ".json");
        std::ofstream output(output_path);
        if (!output) {
            throw std::runtime_error("failed to open output file: " + output_path.string());
        }
        write_json(output, params);
    }
}

int main(int argc, char** argv) {
    try {
        const ProgramOptions options = parse_args(argc, argv);
        write_samples(options.num, options.seed, options.output_dir);
        std::cout << "wrote " << options.num << " parameter files to " << options.output_dir << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
