# raptor_dynamics_sampler

Standalone Python package for generating quadrotor dynamics parameter JSON files.

This package was written by reading RAPTOR's parameter sampler:

`raptor/rl-tools/src/foundation_policy/pre_training/sample_dynamics_parameters.cpp`

It also mirrors the core `rl_tools` L2F sampling order: sample thrust-to-weight,
sample mass through a uniform size distribution, scale thrust curves, sample a
torque-to-inertia ratio, scale arm length and inertia, then sample rotor torque
constants and motor rising/falling time constants.

The `raptor` package is only used as a reference and is not modified.

## Files

- `scripts/sample_dynamics_parameters.py`: command-line generator.
- `package.xml` and `CMakeLists.txt`: minimal catkin package metadata.

## Usage

Generate 10 JSON files:

```bash
cd /home/xin/generate_ws/src
python3 raptor_dynamics_sampler/scripts/sample_dynamics_parameters.py \
  --num 10 \
  --seed 0 \
  --output-dir /tmp/raptor_dynamics_test
```

The output files will be:

```text
/tmp/raptor_dynamics_test/0.json
/tmp/raptor_dynamics_test/1.json
...
/tmp/raptor_dynamics_test/9.json
```

Generate the default 1000 files:

```bash
cd /home/xin/generate_ws/src
python3 raptor_dynamics_sampler/scripts/sample_dynamics_parameters.py \
  --output-dir /home/xin/generate_ws/src/raptor_dynamics_sampler/dynamics_parameters
```

Equivalent explicit form:

```bash
python3 raptor_dynamics_sampler/scripts/sample_dynamics_parameters.py \
  --num 1000 \
  --seed 0 \
  --output-dir /home/xin/generate_ws/src/raptor_dynamics_sampler/dynamics_parameters
```

Each output JSON contains RAPTOR-style fields such as `dynamics.mass`,
`dynamics.rotor_positions`, `dynamics.rotor_thrust_coefficients`,
`dynamics.rotor_time_constants_rising`, `dynamics.rotor_time_constants_falling`,
`dynamics.rotor_torque_constants`, `dynamics.J`, `dynamics.J_inv`,
`integration`, `mdp`, `disturbances`, `domain_randomization`, and `trajectory`.
