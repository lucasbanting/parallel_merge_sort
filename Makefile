CXX=mpicxx

default: all

all: main.cpp
	$(CXX) -o main.out main.cpp