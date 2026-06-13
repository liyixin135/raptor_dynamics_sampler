cmake_minimum_required(VERSION 3.0.2)
project(raptor_dynamics_sampler)

# 这个包只包含 catkin 元数据和一个 Python 可执行脚本，保持依赖尽量轻量。
find_package(catkin REQUIRED)

catkin_package()

# 构建工作空间时，把采样脚本安装到 catkin 包的可执行文件目录。
catkin_install_python(PROGRAMS
  scripts/sample_dynamics_parameters.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
