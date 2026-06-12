cmake_minimum_required(VERSION 3.0.2)
project(raptor_dynamics_sampler)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/sample_dynamics_parameters.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
