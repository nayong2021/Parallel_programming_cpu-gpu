CC=g++
CFLAGS=-pthread
TARGET=glife
TARGET_SOURCE=glife.cpp
nprocs=10
display=0
generation=100

all: $(TARGET)

$(TARGET):
	$(CC) -o $(TARGET) $(TARGET_SOURCE) $(CFLAGS)

test1: $(TARGET)
	./$(TARGET) ./sample_inputs/glider $(display) $(nprocs) 10 10 10

test2: $(TARGET)
	./$(TARGET) ./sample_inputs/make-a_71_81 $(display) $(nprocs) 100 100 100

test3: $(TARGET)
	./$(TARGET) ./sample_inputs/23334m_4505_1008 $(display) $(nprocs) 100 1000 1000

