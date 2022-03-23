Interleaved: 
	ispc sinx_interleaved.ispc -o sinx_interleaved_ispc.o
	ispc sinx_interleaved.ispc -o sinx_interleaved_ispc.asm --emit-asm
	ispc sinx_interleaved.ispc -h sinx_ispc.h
	g++ -c sinx_main.cpp -o sinx_main.o
	g++ sinx_main.o sinx_interleaved_ispc.o -o sinx_interleaved
	./sinx_interleaved
Blocked:
	ispc sinx_blocked.ispc -o sinx_blocked_ispc.o
	ispc sinx_blocked.ispc -o sinx_blocked_ispc.asm --emit-asm
	ispc sinx_blocked.ispc -h sinx_ispc.h
	g++ -c sinx_main.cpp -o sinx_main.o
	g++ sinx_main.o sinx_blocked_ispc.o -o sinx_blocked
	./sinx_blocked
minmax: 
	ispc min_max.ispc -o min_max_ispc.o
	ispc min_max.ispc -o min_max_ispc.asm --emit-asm
	ispc min_max.ispc -h min_max_ispc.h
	g++ -c min_max_main.cpp -o min_max_main.o
	g++ min_max_main.o min_max_ispc.o -o min_max
	./min_max
clean:
	rm *.o *.h *.asm min_max sinx_blocked sinx_interleaved

