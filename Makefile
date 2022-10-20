CXX=mpicxx
CXXFLAGS=-mtune=native -O2

default: all

clean:
	rm -rf main.out
	
all: main.cpp
	$(CXX) $(CXXFLAGS) -o main.out main.cpp