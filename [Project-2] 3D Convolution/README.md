src폴더에서 make 명령어를 통해 gpu, single thread avx, multiple thread avx를 수행하는 실행파일을 컴파일 할 수 있다.

또는 
make 3dconvolution_gpu, make 3dconvolution_avx_st, make 3dconvolution_avx_mult 명령어로 각각의 실행파일을 컴파일 할 수도 있다.

실행파일 목록
3dconvolution_gpu : gpu execution 실행파일
3dconvolution_avx_st : single thread avx execution 실행파일
3dconvolution_avx_mult : multiple thread avx execution 실행파일

input, kernel, output은 실행파일을 실행할 때
./실행파일명 input파일경로 kernel파열경로 output파일경로를
로 실행하여 argument를 입력할 수 있다.

새로 컴파일 하고자 할 땐 make clean명령어로 실행파일을 삭제할 수 있다.

sample data로 test하고자 할 땐

make test1 : sample/test1경로의 데이터로 세 실행파일에서 execution time을test하는 명령어
make test2 : sample/test2경로의 데이터로 세 실행파일에서 execution time을test하는 명령어
make test3 : sample/test3경로의 데이터로 세 실행파일에서 execution time을test하는 명령어
make test4 : sample/test4경로의 데이터로 세 실행파일에서 execution time을test하는 명령어
make test5 : sample/test5경로의 데이터로 세 실행파일에서 execution time을test하는 명령어

make test_all: test1에서 test5까지 순차적으로 수행하는 명령어

로 실행시간을 테스트 할 수 있다.
