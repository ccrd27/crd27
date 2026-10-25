cmake_minimum_required(VERSION 3.16)

project(Research11RCD VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(eigen_headers INTERFACE)
target_include_directories(eigen_headers INTERFACE
    ${PROJECT_SOURCE_DIR}/eigen-3.4.1
)

add_library(amgcl_headers INTERFACE)
target_include_directories(amgcl_headers SYSTEM INTERFACE
    ${PROJECT_SOURCE_DIR}/amgcl
)
target_compile_definitions(amgcl_headers INTERFACE
    AMGCL_NO_BOOST
)

add_library(rcd_core
    src/attack_oracle.cpp
    src/baselines.cpp
    src/certificate.cpp
    src/conflict.cpp
    src/config.cpp
    src/crd.cpp
    src/crd_linear.cpp
    src/defense_oracle.cpp
    src/evaluation.cpp
    src/graph.cpp
    src/objective.cpp
    src/propagation.cpp
    src/progress.cpp
    src/results_writer.cpp
    src/rr_sets.cpp
)

target_include_directories(rcd_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(rcd_core PUBLIC
    eigen_headers
    amgcl_headers
)

if(MSVC)
    target_compile_options(rcd_core PRIVATE /W4)
else()
    target_compile_options(rcd_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(crd
    apps/crd.cpp
)

target_link_libraries(crd PRIVATE
    rcd_core
)
