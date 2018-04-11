# Makefile with support for Windows (mingw32) and NIX (clang / gcc)

CC = gcc

NAME = tinn

SRCS =
SRCS += test.c
SRCS += Tinn.c

# CompSpec defined in windows environment.

ifdef ComSpec
	BIN = $(NAME).exe
else
	BIN = $(NAME)
endif

CFLAGS =
CFLAGS += -std=c99
CFLAGS += -Wshadow -Wall -Wpedantic -Wextra -Wdouble-promotion -Wunused-result
CFLAGS += -g
CFLAGS += -Ofast -march=native -pipe
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

_SYS=$(shell uname -o)
ifeq ($(_SYS),Msys)
	CFLAGS += -posix	# Msys2 / Mingw64 seems to need this for printf / scanf %a support. Sigh..
endif

# Link.

$(BIN): $(SRCS:.c=.o)
	$(CC) $(CFLAGS) $(SRCS:.c=.o) $(LDFLAGS) -o $(BIN)

# Compile.

%.o : %.c Makefile
	$(CC) $(CFLAGS) -MMD -MP -MT $@ -MF $*.td -c $<
	$(RM) $*.d
	$(MV) $*.td $*.d
%.d: ;
-include *.d

clean:
	$(RM) vgcore.*
	$(RM) cachegrind.out.*
	$(RM) callgrind.out.*
	$(RM) saved.tinn
	$(RM) $(BIN)
	$(RM) $(SRCS:.c=.o)
	$(RM) $(SRCS:.c=.d)
