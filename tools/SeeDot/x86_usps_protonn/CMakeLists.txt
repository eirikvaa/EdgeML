cmake_minimum_required(VERSION 3.10.2)
project(x86_protonn_usps)
set(CMAKE_CXX_FLAGS "-L/usr/local/Cellar/llvm/9.0.0_1/lib/ -fopenmp")
set(CMAKE_CXX_COMPILER clang++)

add_executable(
		main
		main.cpp
		datatypes.h
		library.h
		library.cpp
		predictors.h
		seedot_fixed_model.h
		seedot_fixed.cpp
		profile.h
		profile.cpp
		protonn_float_model.h
		protonn_float.cpp
		bonsai_float_model.h
		bonsai_float.cpp
)