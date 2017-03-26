LIBS=-lGL -lGLU -lglut -lGLEW
CFLAGS=-g -O0 -Wall -Wextra
CXXFLAGS=-std=c++11 -pedantic
LDFLAGS=

run: grafhazi
	LIBGL_ALWAYS_SOFTWARE=1 ./grafhazi

debug: grafhazi
	LIBGL_ALWAYS_SOFTWARE=1 gdb ./grafhazi

all: grafhazi

.PHONY: clean all run

clean:
	rm -f grafhazi main.o

grafhazi: main.o
	g++ -o grafhazi ${LIBS} ${LDFLAGS} main.o
	
main.o: main.cpp
	g++ -c -o main.o ${CFLAGS} ${CXXFLAGS} main.cpp

