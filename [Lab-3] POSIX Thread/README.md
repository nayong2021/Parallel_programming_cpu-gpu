make all: glife.cpp에서 glife실행파일을 컴파일 하는 명령어

make test1: glider한개가 포함된 sample을 10*10 크기의 grid로 load하여 테스트 하는 명령어
기본값으로 nprocs = 10, display = 0이며 10번의 generation동안 실행된다.

make test2: make-a_71_81샘플을 100*100크기 grid로 load하여 테스트 하는 명령어
기본값으로 nprocs = 10, display = 0이며 100번의 generation동안 실행된다.

make test3: 23334m_4505_1008샘플을 1000*1000크기 grid로 load하여 테스트 하는 명령어
기본값으로 nprocs = 10, display = 0이며 100번의 generation동안 실행된다.

test1, 2, 3 명령어의 경우 nprocs=i display=0 or 1 로 파라미터를 변경할 수 있다.
ex) test2에서 serial하게 실행하고자 nprocs에 0을 넣고자 할 때의 명령어
=> make test2 nprocs=0

위의 Makefile을 이용한 test명령어들이 아닌 컴파일은 make all을 통해 하고
./glife <input file> <display> <nprocs> <# of generations> <width> <height>
위와 같은 형태로 직접 실행해도 원하는 파라미터를 사용해서 정상적으로 실행할 수 있다.

