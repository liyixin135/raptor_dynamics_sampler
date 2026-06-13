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
//
// 目标只做一件事：随机生成四旋翼动力学参数 JSON 文件。
// 它不包含 teacher policy 训练，不包含 student policy 蒸馏，也不生成
// foundation policy 的整套最终参数。
//
// 设计原则：
// 1. 只使用 C++17 标准库，不依赖 RAPTOR/rl_tools 头文件。
// 2. 先从一组 Crazyflie 风格的“名义参数”开始。
// 3. 按 RAPTOR/Python 版本的顺序随机采样质量、推重比、惯量、电机参数等。
// 4. 最后把已经采样好的确定性参数写成 JSON，并把 domain_randomization 清零。
//
// 推荐编译命令：
// g++ -std=c++17 -O2 cpp/sample_dynamics_parameters.cpp -o sample_dynamics_parameters_cpp

namespace fs = std::filesystem;

// 四旋翼固定有 4 个电机/旋翼。
constexpr int N_ROTORS = 4;
constexpr double PI = 3.14159265358979323846;

// Vec3 表示三维向量，例如位置 [x, y, z]、重力方向、推力方向。
using Vec3 = std::array<double, 3>;

// Mat3 表示 3x3 矩阵，这里主要用于机体惯量矩阵 J 和它的逆 J_inv。
using Mat3 = std::array<std::array<double, 3>, 3>;

// 每个电机的推力曲线有 3 个系数：
// thrust = c0 + c1 * u + c2 * u^2，其中 u 是归一化电机命令。
using RotorCoefficients = std::array<double, 3>;

struct ActionLimit {
    // 归一化动作范围。这里和 Python 版本一致，电机命令 u 属于 [0, 1]。
    double min = 0.0;
    double max = 1.0;
};

struct Dynamics {
    // 四个旋翼在机体系下的位置，单位是米。
    // 机体系可以理解为固定在机身上的坐标系。
    std::array<Vec3, N_ROTORS> rotor_positions{};

    // 每个旋翼正推力方向。这里四个电机都沿机体系 +z 方向产生推力。
    std::array<Vec3, N_ROTORS> rotor_thrust_directions{};

    // 每个旋翼反扭矩方向。正反桨交替，所以 z 方向符号交替。
    std::array<Vec3, N_ROTORS> rotor_torque_directions{};

    // 四个旋翼的推力曲线系数，公式是 thrust = c0 + c1*u + c2*u^2。
    std::array<RotorCoefficients, N_ROTORS> rotor_thrust_coefficients{};

    // 旋翼阻力矩常数。RAPTOR 的模型里它把推力换算成绕 z 轴的反扭矩。
    std::array<double, N_ROTORS> rotor_torque_constants{};

    // 电机一阶响应时间常数：命令增大时使用 rising，命令减小时使用 falling。
    // 数值越大，电机响应越慢。
    std::array<double, N_ROTORS> rotor_time_constants_rising{};
    std::array<double, N_ROTORS> rotor_time_constants_falling{};

    // 飞机质量，单位 kg。
    double mass = 0.0;

    // 世界系重力向量，单位 m/s^2。
    Vec3 gravity{};

    // 机体惯量矩阵和它的逆，单位分别是 kg*m^2 和 1/(kg*m^2)。
    // 本程序只缩放对角线项，保持和 Python 版本一致。
    Mat3 J{};
    Mat3 J_inv{};

    // 悬停油门占动作范围的比例。采样并缩放推力曲线后需要重新计算。
    double hovering_throttle_relative = 0.0;

    ActionLimit action_limit{};
};

struct RangeConfig {
    // 以下字段就是 RAPTOR/Python 版本使用的随机采样范围。
    // min/max 成对出现的字段使用均匀分布采样。
    double thrust_to_weight_min = 1.5;
    double thrust_to_weight_max = 5.0;

    // torque_to_inertia 越大，代表同样机体能产生更大的角加速度。
    double torque_to_inertia_min = 40.0;
    double torque_to_inertia_max = 1200.0;

    // 质量范围。注意实际采样不是直接均匀采样 mass，而是均匀采样 size，再令 mass=size^3。
    double mass_min = 0.02;
    double mass_max = 5.0;

    // 机臂尺寸额外随机偏差，用于让同样质量下的机体尺寸也有变化。
    double mass_size_deviation = 0.1;

    // 电机响应时间常数范围，单位秒。
    double rotor_time_constant_rising_min = 0.03;
    double rotor_time_constant_rising_max = 0.10;
    double rotor_time_constant_falling_min = 0.03;
    double rotor_time_constant_falling_max = 0.30;

    // 推力到反扭矩的换算常数范围。
    double rotor_torque_constant_min = 0.005;
    double rotor_torque_constant_max = 0.05;

    // Python 版本里保留了这个字段，但当前范围是 0，所以不会产生姿态安装偏差。
    double orientation_offset_angle_max = 0.0;

    // 扰动力强度上限。最终 std 会结合推重比和质量继续缩放。
    double disturbance_force_max = 0.3;
};

struct InitConfig {
    // 这里保存 MDP 初始状态采样相关配置。
    // 本程序只会根据飞机尺寸更新 max_position，其余保持 Python 版本默认值。
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
    // 训练环境终止条件。这里只根据飞机尺寸更新 position_threshold。
    bool enabled = true;
    double position_threshold = 1.0;
    double linear_velocity_threshold = 2.0;
    double angular_velocity_threshold = 35.0;
    double position_integral_threshold = 10000.0;
    double orientation_integral_threshold = 50000.0;
};

struct Disturbance {
    // 扰动用正态分布描述：mean 是均值，std 是标准差。
    // 当前只随机化 random_force.std，random_torque 保持 0。
    double mean = 0.0;
    double std = 0.0;
};

struct Parameters {
    // 这个结构体对应最终 JSON 的主要内容。
    // 为了让代码更容易读，reward、noise、trajectory 等固定字段没有单独建结构体，
    // 而是在 write_json() 里直接写出固定值。
    Dynamics dynamics{};
    InitConfig init{};
    TerminationConfig termination{};
    Disturbance random_force{};
    Disturbance random_torque{};
    RangeConfig domain_randomization{};
};

struct ProgramOptions {
    // 命令行参数的解析结果。
    // --num        生成多少个 JSON 文件。
    // --seed       随机数种子，相同 seed 会产生可复现的随机序列。
    // --output-dir 输出目录。
    int num = 1000;
    unsigned int seed = 0;
    fs::path output_dir = "dynamics_parameters";
};

double vector_norm(const Vec3& values) {
    // 三维向量欧氏范数：sqrt(x^2 + y^2 + z^2)。
    // 用于计算 |gravity| 和电机到机体中心的距离。
    return std::sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
}

double uniform(std::mt19937& rng, double min_value, double max_value) {
    // 从 [min_value, max_value] 均匀采样一个 double。
    std::uniform_real_distribution<double> distribution(min_value, max_value);
    return distribution(rng);
}

double gaussian(std::mt19937& rng, double mean, double stddev) {
    // 从正态分布 N(mean, stddev^2) 采样一个 double。
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(rng);
}

void fill_all(std::array<double, N_ROTORS>& values, double value) {
    // 四个旋翼经常共享同一个采样值，例如同一个 torque_constant。
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
    // 每次采样都会从这份模板重新开始，避免上一次采样影响下一次采样。
    Parameters params;
    Dynamics& d = params.dynamics;

    // rotor_positions 的四行分别是四个电机的位置。
    // 这里是一个 x/y 平面上的正方形布局，z=0 表示电机和机体中心在同一平面。
    d.rotor_positions = {{{{0.028, -0.028, 0.0}},
                          {{-0.028, -0.028, 0.0}},
                          {{-0.028, 0.028, 0.0}},
                          {{0.028, 0.028, 0.0}}}};

    // 推力方向全部朝 +z。这里没有做旋翼安装角随机化。
    d.rotor_thrust_directions = {{{{0.0, 0.0, 1.0}},
                                  {{0.0, 0.0, 1.0}},
                                  {{0.0, 0.0, 1.0}},
                                  {{0.0, 0.0, 1.0}}}};

    // 反扭矩方向交替，模拟相邻电机旋转方向相反。
    d.rotor_torque_directions = {{{{0.0, 0.0, -1.0}},
                                  {{0.0, 0.0, 1.0}},
                                  {{0.0, 0.0, -1.0}},
                                  {{0.0, 0.0, 1.0}}}};

    // 名义推力曲线。后面会整体乘以缩放因子，使最大推力符合采样到的质量和推重比。
    d.rotor_thrust_coefficients = {{{{0.00352526, 0.01437313, 0.09223048}},
                                    {{0.00352526, 0.01437313, 0.09223048}},
                                    {{0.00352526, 0.01437313, 0.09223048}},
                                    {{0.00352526, 0.01437313, 0.09223048}}}};
    fill_all(d.rotor_torque_constants, 4.665e-3);
    fill_all(d.rotor_time_constants_rising, 0.05545454545454546);
    fill_all(d.rotor_time_constants_falling, 0.24939393939393945);

    // 名义质量由几个部件质量相加得到，和 Python 文件一致。
    d.mass = 0.027 + 0.0017 + 0.0003 + 0.0016;

    d.gravity = {{0.0, 0.0, -9.81}};

    // 名义惯量矩阵和逆矩阵。这里都是对角矩阵，非对角项为 0。
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
    // 四个电机最大推力相加，就是整架飞机的最大总推力。
    const double max_action = dynamics.action_limit.max;
    double total = 0.0;
    for (const auto& coeffs : dynamics.rotor_thrust_coefficients) {
        total += coeffs[0] + coeffs[1] * max_action + coeffs[2] * max_action * max_action;
    }
    return total;
}

void update_hovering_throttle(Dynamics& dynamics) {
    // 悬停时四个电机总推力等于重力：sum(thrust_i) = mass * |gravity|。
    // 因为四个电机使用同一条平均推力曲线，所以先计算“单个电机需要承担的悬停推力”。
    const double per_rotor_hover_thrust = dynamics.mass * vector_norm(dynamics.gravity) / N_ROTORS;

    // 对四个电机的 c0/c1/c2 取平均，得到一条平均推力曲线。
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

    // 转成相对比例。当前 action 范围是 [0,1]，所以这里数值等于 throttle 本身；
    // 但保留这个公式可以兼容未来 action_limit 改成其他范围。
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
    // 这个函数是整个程序最重要的部分：生成一组随机动力学参数。
    // 注意：每次调用都先复制名义参数，再在这份副本上做随机缩放。
    Parameters params = nominal_parameters();
    const RangeConfig ranges = raptor_sampling_ranges();
    Dynamics& d = params.dynamics;

    const double gravity_norm = vector_norm(d.gravity);
    const double mass_nominal = d.mass;

    // 名义最大推重比 = 名义最大总推力 / 名义重量。
    // 后面会采样一个目标 thrust_to_weight，并用二者比值缩放推力曲线。
    const double thrust_to_weight_nominal = max_total_thrust(d) / (mass_nominal * gravity_norm);

    // 1. 随机采样质量。为了覆盖大范围质量，先均匀采样“尺寸”，再用 mass = size^3。
    // 这样做的直觉是：如果几何尺寸放大 s 倍，体积和质量大约放大 s^3 倍。
    const double relative_size_min = std::cbrt(ranges.mass_min);
    const double relative_size_max = std::cbrt(ranges.mass_max);
    const double size_new = uniform(rng, relative_size_min, relative_size_max);
    const double mass_new = std::min(std::max(size_new * size_new * size_new, ranges.mass_min), ranges.mass_max);

    // scale_relative 是相对于名义机体的线性尺寸比例。
    // factor_mass 是相对于名义机体的质量比例。
    const double scale_relative = std::cbrt(mass_new / mass_nominal);
    const double factor_mass = mass_new / mass_nominal;
    d.mass = mass_new;

    // 2. 随机采样最大推重比 thrust_to_weight = max_total_thrust / (mass * g)。
    // 推重比越大，飞机最大推力相对自重越强。
    const double thrust_to_weight = uniform(rng, ranges.thrust_to_weight_min, ranges.thrust_to_weight_max);
    const double factor_thrust_to_weight = thrust_to_weight / thrust_to_weight_nominal;

    // 3. 同时考虑新质量和目标推重比，缩放所有电机推力曲线系数。
    // 质量变大，需要更大总推力才能达到同样推重比；目标推重比变大，也需要更大总推力。
    const double factor_thrust_coefficients = factor_thrust_to_weight * factor_mass;
    for (auto& rotor_coeffs : d.rotor_thrust_coefficients) {
        for (double& coeff : rotor_coeffs) {
            coeff *= factor_thrust_coefficients;
        }
    }

    // 4. 随机采样 torque_to_inertia，并通过调整惯量矩阵 J 来匹配目标角加速度能力。
    //
    // 力矩近似来自“电机推力 * 力臂”。
    // 这里 first_rotor_distance_nominal * sqrt(2) 表示从 x/y 单轴距离换算到对角线臂长。
    // torque_to_inertia = max_torque / Jxx，直观上表示最大滚转/俯仰角加速度能力。
    const double max_thrust_per_rotor = thrust_to_weight * d.mass * gravity_norm / N_ROTORS;
    const double first_rotor_distance_nominal = std::abs(d.rotor_positions[0][0]);
    const double max_torque = first_rotor_distance_nominal * std::sqrt(2.0) * max_thrust_per_rotor;
    const double torque_to_inertia_nominal = max_torque / d.J[0][0];
    const double torque_to_inertia = uniform(rng, ranges.torque_to_inertia_min, ranges.torque_to_inertia_max);
    const double torque_to_inertia_factor = torque_to_inertia / torque_to_inertia_nominal;

    // 5. 机臂长度随质量对应的尺寸变化，再叠加 mass_size_deviation 随机偏差。
    //
    // rotor_distance_factor 控制 rotor_positions 的缩放。
    // inertia_factor 控制惯量矩阵的缩放，用来让采样后的 torque_to_inertia 接近目标值。
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
            // x/y/z 坐标都乘同一个比例，保持机体几何形状相似。
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
    //
    // disturbance_multiple 先在 [0, (thrust_to_weight - 1) * max] 里采样。
    // 后面再乘 thrust_to_weight 和 mass，并除以 3，和 Python 版本保持一致。
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
    // JSON 手写输出时用于生成缩进空格。
    return std::string(static_cast<std::size_t>(spaces), ' ');
}

void write_double(std::ostream& os, double value) {
    // setprecision(17) 尽量保留 double 精度，避免写出 JSON 后损失太多有效数字。
    os << std::setprecision(17) << value;
}

void write_vec3(std::ostream& os, const Vec3& values) {
    // 输出形如 [x, y, z] 的 JSON 数组。
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
    // 输出四个电机共享格式的数组，例如 rotor_torque_constants。
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
    // 输出四行 Vec3，例如四个旋翼的位置、推力方向、反扭矩方向。
    os << "[\n";
    for (int i = 0; i < N_ROTORS; ++i) {
        os << indent(spaces + 2);
        write_vec3(os, values[i]);
        os << (i + 1 == N_ROTORS ? "\n" : ",\n");
    }
    os << indent(spaces) << "]";
}

void write_coefficients_array(std::ostream& os, const std::array<RotorCoefficients, N_ROTORS>& values, int spaces) {
    // 输出四个电机的推力曲线系数，每行是 [c0, c1, c2]。
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
    // 输出 3x3 矩阵，例如 J 和 J_inv。
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
    // 输出 domain_randomization 字段。
    // 保存采样结果时这些值应该全是 0，表示以后读取该 JSON 时不要再次随机化。
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
    // 手写 JSON 的原因是本程序只依赖标准库，不引入第三方 JSON 库。
    // 字段顺序尽量跟 Python 版本保持一致，方便对照阅读和 diff。
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
    // 解析命令行参数。支持：
    //   --num 10
    //   --seed 0
    //   --output-dir test_output_cpp
    //
    // 这里使用最简单的手写解析方式，避免引入额外依赖。
    ProgramOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            // 像 --num 这种参数后面必须跟一个值；如果缺值就报错。
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
    // 确保输出目录存在。如果已经存在，不会删除里面的旧文件。
    fs::create_directories(output_dir);

    // mt19937 是 C++ 标准库里的伪随机数生成器。
    // 同一个 seed 会得到同样的随机序列，便于复现实验。
    std::mt19937 rng(seed);

    for (int index = 0; index < num; ++index) {
        // 每个 index 生成一组独立参数，并写到 index.json。
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
        // 主流程：解析参数 -> 采样并写文件 -> 打印结果。
        const ProgramOptions options = parse_args(argc, argv);
        write_samples(options.num, options.seed, options.output_dir);
        std::cout << "wrote " << options.num << " parameter files to " << options.output_dir << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
