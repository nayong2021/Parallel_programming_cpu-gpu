make로 multisrv와 echocli를 complie한 다음
./multisrv -a (acceptor thread 개수) -w (worker thread 개수)로 원하는 스레드 개수를 지정해 실행한 다음
SERVERHOST=localhost ./echocli (PORT넘버)로 연결을 할 수 있다.