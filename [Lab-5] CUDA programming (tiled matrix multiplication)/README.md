make명령으로 non_tiled, tiled 실행파일을 생성할 수 있다.
또는 make non_tiled, make tiled로 각각을 별개로 build할 수 있다.
make test_non_tiled : non_tiled를 build하고 실행한다.
make test_tiled : tiled를 build하고 실행한다.
make clean : executable file들을 삭제한다.
tiled, non_tiled는 실행했을 시 M, N 행렬의 행, 열 길이를 
M_xlen, M_ylen, N_xlen, N_ylen 순으로 입력받는다.
그 다음 랜덤하게 matrix 두 개를 생성한 수 matrix multiplication을 수행한 후
경과된 시간을 출력해준다.
tile의 크기는 code내부에서 #define TILE_WIDTH로에 정의되어 있다.
