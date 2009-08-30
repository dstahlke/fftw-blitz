CFLAGS_O = -g -DBZ_DEBUG -O0
#CFLAGS_O = -g -O3
CXXFLAGS = -Wall -I. `pkg-config blitz fftw3 --cflags` $(CFLAGS_O)
LIBRARIES = `pkg-config blitz fftw3 --libs`
EXEC = demo

OBJECTS = fftw_blitz.o demo.o

all: $(EXEC)

$(EXEC): $(OBJECTS)
	g++ -o $@ $(OBJECTS) $(LIBRARIES)

clean:
	rm -f $(OBJECTS) $(EXEC)
