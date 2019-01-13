cmake_minimum_required(VERSION 3.13)
project(NewTpcCalibration)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_PREFIX_PATH $ENV{ROOTSYS} ${CMAKE_PREFIX_PATH})

find_package(ROOT REQUIRED)

include(${ROOT_USE_FILE})

ROOT_GENERATE_DICTIONARY(
        G__${PROJECT_NAME} types.h LINKDEF ${PROJECT_NAME}LinkDef.h
)

add_executable(NewTpcCalibration main.cpp G__${PROJECT_NAME}.cxx)
target_link_libraries(NewTpcCalibration PUBLIC ${ROOT_LIBRARIES} ROOT::MathMore)