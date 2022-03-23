CC=nvcc

all : non_tiled tiled

non_tiled : non_tiled.cu
	$(CC) non_tiled.cu -o non_tiled

tiled : tiled.cu
	$(CC) tiled.cu -o tiled

test_tiled : tiled
	./tiled

test_non_tiled : non_tiled
	./non_tiled

clean : 
	rm tiled non_tiled
