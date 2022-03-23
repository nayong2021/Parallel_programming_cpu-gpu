CC=nvcc
CFLAGS=-o
TARGET=2dconvolution
TARGET_SOURCE=2dconvolution.cu

all:$(TARGET)

clean:
	rm $(TARGET)

$(TARGET):$(TARGET_SOURCE)
	$(CC) $(TARGET_SOURCE) $(CFLAGS) $(TARGET)
