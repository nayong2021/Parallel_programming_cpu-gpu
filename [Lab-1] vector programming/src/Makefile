CC=gcc
CFLAGS=-mavx2
TARGET=matmul
TARGET_SOURCE=main.c

all :
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET_SOURCE)

test1 :
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET_SOURCE)
	./$(TARGET) -v 1

test2 :
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET_SOURCE)
	./$(TARGET) -v 2
