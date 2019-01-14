cmake_minimum_required(VERSION 3.0)
project(NewTpcCalibration)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH $ENV{ROOTSYS} ${CMAKE_PREFIX_PATH})

find_package(ROOT REQUIRED)

include(${ROOT_USE_FILE})

add_executable(NewTpcCalibration main.cpp)
target_link_libraries(NewTpcCalibration PUBLIC ${ROOT_LIBRARIES} ROOT::MathMore)

add_executable(vDCalibration vDCalibration.cpp)
target_link_libraries(vDCalibration PUBLIC ${ROOT_LIBRARIES} ROOT::MathMore)
