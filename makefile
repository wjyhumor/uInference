
VPATH=./src/
OBJDIR=./obj/
EXEC=uInference.bin

CC=gcc
LDFLAGS=-lm
CFLAGS=-Wall -Wfatal-errors -Ofast

OBJ=common.o main.o 

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) 

all: obj $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)

