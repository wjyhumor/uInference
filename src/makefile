USE_VOT_TRAX=0

CC=gcc
CXX=g++

LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm
CXXFLAGS= -Wall `pkg-config --cflags opencv` -std=c++0x -lstdc++ -O3 -fPIC
HEADERS = $(wildcard *.h) $(wildcard *.hpp)
TARGET_LIB = libuinference.so
OBJS = 

ifeq ($(USE_VOT_TRAX), 1)
CXXFLAGS+= -ltrax
LDFLAGS+= -ltrax 
endif

ALL = uinference.bin $(TARGET_LIB)

all: $(ALL)

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -shared -o $@ $^

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean

clean:
	rm -rf *.o *.bin *.so
