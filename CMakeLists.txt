cmake_minimum_required(VERSION 3.0.2)
project(raptor_dynamics_sampler)

# This package is intentionally lightweight: catkin metadata plus one Python executable.
find_package(catkin REQUIRED)

catkin_package()

# Install the sampler into the catkin package bin destination when the workspace is built.
catkin_install_python(PROGRAMS
  scripts/sample_dynamics_parameters.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
