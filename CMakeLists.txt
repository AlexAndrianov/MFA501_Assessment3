cmake_minimum_required(VERSION 3.5)

project(Assesment_3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Assesment_3 main.cpp
    Matrix.h
    NeuralNetwork.h
    Population.h)
