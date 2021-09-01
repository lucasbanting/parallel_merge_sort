CXX=mpicxx

default: all

clean:
	rm -rf main.out
	
all: main.cpp
	$(CXX) -O2 -o main.out main.cpp