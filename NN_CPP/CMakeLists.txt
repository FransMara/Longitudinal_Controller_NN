cmake_minimum_required(VERSION 3.23)
project(NN_CPP)

set(CMAKE_CXX_STANDARD 14)

add_executable(NN_CPP neural_network_test.cc)

configure_file(test_data/velocity.csv ${CMAKE_BINARY_DIR}/velocity.csv COPYONLY)
configure_file(test_data/acceleration.csv ${CMAKE_BINARY_DIR}/acceleration.csv COPYONLY)

include_directories(${CMAKE_BINARY_DIR})





