cmake_minimum_required(VERSION 3.10)
project(raptor_dynamics_sampler)

set(CATKIN_ENABLE_TESTING OFF CACHE BOOL "Disable catkin testing for this package" FORCE)

find_package(catkin REQUIRED)

catkin_package()

add_executable(sample_dynamics_parameters_cpp
  cpp/sample_dynamics_parameters.cpp
)
target_compile_features(sample_dynamics_parameters_cpp PRIVATE cxx_std_17)

catkin_install_python(PROGRAMS
  scripts/sample_dynamics_parameters.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS sample_dynamics_parameters_cpp
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
