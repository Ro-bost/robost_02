cmake_minimum_required(VERSION 3.16)
project(robost_mpc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(ROBOST_MPC_WITH_VIEWER "Build the GLFW live viewer" ON)
include(FetchContent)

# --- Eigen3 (header only, apt: libeigen3-dev) --------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# --- MuJoCo: reuse the library shipped in the `robost` conda env's pip wheel --
if(NOT MUJOCO_DIR)
  if(DEFINED ENV{CONDA_PREFIX})
    set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}")
  endif()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
    OUTPUT_VARIABLE MUJOCO_DIR OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _mj_rc)
  if(NOT _mj_rc EQUAL 0)
    message(FATAL_ERROR "Python 'mujoco' not found. Run `conda activate robost` or pass -DMUJOCO_DIR=...")
  endif()
endif()
file(GLOB MUJOCO_LIBRARY "${MUJOCO_DIR}/libmujoco.so*")
if(NOT MUJOCO_LIBRARY OR NOT EXISTS "${MUJOCO_DIR}/include/mujoco/mujoco.h")
  message(FATAL_ERROR "libmujoco / headers not found under ${MUJOCO_DIR}")
endif()
list(GET MUJOCO_LIBRARY 0 MUJOCO_LIBRARY)
add_library(mujoco::mujoco SHARED IMPORTED)
set_target_properties(mujoco::mujoco PROPERTIES
  IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_DIR}/include")
message(STATUS "MuJoCo: ${MUJOCO_LIBRARY}")

# --- OSQP: system install if present, otherwise build v1.0.0 from source ------
find_package(osqp QUIET)
if(TARGET osqp::osqpstatic)
  set(OSQP_TARGET osqp::osqpstatic)
elseif(TARGET osqp::osqp)
  set(OSQP_TARGET osqp::osqp)
else()
  set(OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_DEMO_EXE OFF CACHE BOOL "" FORCE)
  set(OSQP_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(osqp
    GIT_REPOSITORY https://github.com/osqp/osqp.git
    GIT_TAG v1.0.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(osqp)
  set(OSQP_TARGET osqpstatic)
endif()

# --- Controller library ---------------------------------------------------------
add_library(robost_mpc_core
  src/convex_mpc.cpp
  src/mujoco_robot.cpp
  src/locomotion_controller.cpp)
target_include_directories(robost_mpc_core PUBLIC include)
target_link_libraries(robost_mpc_core PUBLIC Eigen3::Eigen mujoco::mujoco ${OSQP_TARGET})
target_compile_options(robost_mpc_core PRIVATE -Wall -Wextra)

# --- Simulation executable --------------------------------------------------------
add_executable(rs06_mpc src/main.cpp)
target_link_libraries(rs06_mpc PRIVATE robost_mpc_core)
set_target_properties(rs06_mpc PROPERTIES BUILD_RPATH "${MUJOCO_DIR}")

if(ROBOST_MPC_WITH_VIEWER)
  find_package(glfw3 QUIET)
  if(TARGET glfw)
    set(GLFW_TARGET glfw)
  else()
    # Runtime libglfw.so.3 is installed but libglfw3-dev may not be: use the
    # matching public header only and link the runtime library directly.
    find_library(GLFW_LIBRARY NAMES glfw libglfw.so.3)
    set(GLFW_HEADER "${CMAKE_BINARY_DIR}/glfw/include/GLFW/glfw3.h")
    if(GLFW_LIBRARY AND NOT EXISTS "${GLFW_HEADER}")
      file(DOWNLOAD https://raw.githubusercontent.com/glfw/glfw/3.3.10/include/GLFW/glfw3.h
           "${GLFW_HEADER}" STATUS _dl)
    endif()
    if(GLFW_LIBRARY AND EXISTS "${GLFW_HEADER}")
      add_library(glfw_runtime INTERFACE)
      target_include_directories(glfw_runtime INTERFACE "${CMAKE_BINARY_DIR}/glfw/include")
      target_link_libraries(glfw_runtime INTERFACE "${GLFW_LIBRARY}")
      set(GLFW_TARGET glfw_runtime)
    endif()
  endif()
  if(GLFW_TARGET)
    target_link_libraries(rs06_mpc PRIVATE ${GLFW_TARGET})
    target_compile_definitions(rs06_mpc PRIVATE ROBOST_MPC_WITH_VIEWER)
    message(STATUS "Viewer: enabled")
  else()
    message(WARNING "GLFW not found: building headless only")
  endif()
endif()
