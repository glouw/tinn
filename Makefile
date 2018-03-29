CC = gcc

NAME = tinn

SRCS =
SRCS += main.c
SRCS += Tinn.c

# CompSpec defined in windows environment.
ifdef ComSpec
	BIN = $(NAME).exe
else
	BIN = $(NAME)
endif

CFLAGS =
CFLAGS += -std=c89
CFLAGS += -Wshadow -Wall -Wpedantic -Wextra -Wdouble-promotion -Wunused-result
CFLAGS += -g
CFLAGS += -O2 -march=native -pipe
CFLAGS += -flto

LDFLAGS =
LDFLAGS += -lm

ifdef ComSpec
	RM = del /F /Q
	MV = ren
else
	RM = rm -f
	MV = mv -f
endif

# Link.
$(BIN): $(SRCS:.c=.o)
	echo $(CC) *.o -o $(BIN)
	$(CC) $(CFLAGS) $(SRCS:.c=.o) $(LDFLAGS) -o $(BIN)

# Compile.
%.o : %.c Makefile
	echo $(CC) -c $*.c
	$(CC) $(CFLAGS) -MMD -MP -MT $@ -MF $*.td -c $<
	$(RM) $*.d
	$(MV) $*.td $*.d
%.d: ;
-include *.d

clean:
	$(RM) vgcore.*
	$(RM) cachegrind.out.*
	$(RM) callgrind.out.*
	$(RM) $(BIN)
	$(RM) $(SRCS:.c=.o)
	$(RM) $(SRCS:.c=.d)
