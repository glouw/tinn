# Makefile with support for Windows (mingw32) and NIX (clang / gcc)

CC=g++
CFLAGS=-c
LDFLAGS=
SOURCES=test.cpp Tinn.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=tinn

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *o 


