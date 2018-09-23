BIN = test

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto 

LDFLAGS = -lm

CC = gcc

SRC = test.c Tinn.c

all:
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRC) -o $(BIN)

run:
	./$(BIN)

clean:
	rm -f $(BIN)
