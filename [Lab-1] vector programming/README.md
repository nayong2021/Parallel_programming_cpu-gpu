main.c : main source code
matmul : executive file generated from main.c by Makefile

you can test scalar and vector multiplication
by Makefile command or excute matmul by your self and pass v argument (1 :  vector matrix multiplication, 2 : scalar matrix multiplication)
ex : ./matmul -v 1 (vector matrix multiplication test)

Makefile command
make : generate executive matmul file from main.c
make all : same as make
make test1 : generate executive matmul file from main.c and test vector version matrix multiplication
make test2 : generate executive matmul file from main.c and test scalar version matrix multiplication
