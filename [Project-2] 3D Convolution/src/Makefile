all: 3dconvolution_gpu 3dconvolution_avx_st 3dconvolution_avx_mult

3dconvolution_gpu : 
	nvcc 3dconvolution_gpu.cu -o 3dconvolution_gpu

3dconvolution_avx_st : 
	g++ -mavx2 -mfma 3dconvolution_avx_st.cpp -o 3dconvolution_avx_st

3dconvolution_avx_mult : 
	g++ -mavx2 -mfma 3dconvolution_avx_mult.cpp -o 3dconvolution_avx_mult -pthread

clean :
	rm 3dconvolution_gpu 3dconvolution_avx_st 3dconvolution_avx_mult

test_all: test1 test2 test3 test4 test5

test1: 3dconvolution_gpu 3dconvolution_avx_st 3dconvolution_avx_mult
	echo test1
	./3dconvolution_gpu sample/test1/input.txt sample/test1/kernel.txt sample/test1/output.txt
	./3dconvolution_avx_st sample/test1/input.txt sample/test1/kernel.txt sample/test1/output.txt
	./3dconvolution_avx_mult sample/test1/input.txt sample/test1/kernel.txt sample/test1/output.txt

test2: 3dconvolution_gpu 3dconvolution_avx_st 3dconvolution_avx_mult
	echo test2
	./3dconvolution_gpu sample/test2/input.txt sample/test2/kernel.txt sample/test2/output.txt
	./3dconvolution_avx_st sample/test2/input.txt sample/test2/kernel.txt sample/test2/output.txt
	./3dconvolution_avx_mult sample/test2/input.txt sample/test2/kernel.txt sample/test2/output.txt

test3: 3dconvolution_gpu 3dconvolution_avx_st 3dconvolution_avx_mult
	echo test3
	./3dconvolution_gpu sample/test3/input.txt sample/test3/kernel.txt sample/test3/output.txt
	./3dconvolution_avx_st sample/test3/input.txt sample/test3/kernel.txt sample/test3/output.txt
	./3dconvolution_avx_mult sample/test3/input.txt sample/test3/kernel.txt sample/test3/output.txt

test4: 3dconvolution_gpu 3dconvolution_avx_st 3dconvolution_avx_mult
	echo test4
	./3dconvolution_gpu sample/test4/input.txt sample/test4/kernel.txt sample/test4/output.txt
	./3dconvolution_avx_st sample/test4/input.txt sample/test4/kernel.txt sample/test4/output.txt
	./3dconvolution_avx_mult sample/test4/input.txt sample/test4/kernel.txt sample/test4/output.txt

test5: 3dconvolution_gpu 3dconvolution_avx_st 3dconvolution_avx_mult
	echo test5
	./3dconvolution_gpu sample/test5/input.txt sample/test5/kernel.txt sample/test5/output.txt
	./3dconvolution_avx_st sample/test5/input.txt sample/test5/kernel.txt sample/test5/output.txt
	./3dconvolution_avx_mult sample/test5/input.txt sample/test5/kernel.txt sample/test5/output.txt
