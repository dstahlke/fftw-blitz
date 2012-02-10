#	Copyright 2010 Daniel Stahlke
#
#	This file is part of fftw-blitz.
#	
#	fftw-blitz is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#	
#	fftw-blitz is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#	
#	You should have received a copy of the GNU General Public License
#	along with fftw-blitz.  If not, see <http://www.gnu.org/licenses/>.

CFLAGS_O = -g -DBZ_DEBUG -O0
#CFLAGS_O = -g -O3
CXXFLAGS = -Wall -I. `pkg-config blitz fftw3 --cflags` $(CFLAGS_O)
LIBRARIES = `pkg-config blitz fftw3 --libs`

#CXXFLAGS += -DFFTWBLITZ_MT=1
#LIBRARIES += -lboost_thread-mt

EXEC = demo

OBJECTS = fftw_blitz.o demo.o

all: $(EXEC)

$(EXEC): $(OBJECTS)
	g++ -o $@ $(OBJECTS) $(LIBRARIES)

clean:
	rm -f $(OBJECTS) $(EXEC)

cppcheck:
	cppcheck --template gcc --enable=all -q *.h .
