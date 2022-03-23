/* file: echosrv.c

   Bare-bones single-threaded TCP server. Listens for connections
   on "ephemeral" socket, assigned dynamically by the OS.

   This started out with an example in W. Richard Stevens' book
   "Advanced Programming in the Unix Environment".  I have
   modified it quite a bit, including changes to make use of my
   own re-entrant version of functions in echolib.

   NOTE: See comments starting with "NOTE:" for indications of
   places where code needs to be added to make this multithreaded.
   Remove those comments from your solution before turning it in,
   and replace them by comments explaining the parts you have
   changed.

   Ted Baker
   February 2015

 */

#include "config.h"
/* not needed now, but will be needed in multi-threaded version */
#include "pthread.h"
#include "echolib.h"
#include "checks.h"
#include <unistd.h>

#define FALSE (0)
#define TRUE (1)

typedef struct _work {
  connection_t *conn;
  int number;
}work;

pthread_mutex_t conn_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t work_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t idle_acceptor = PTHREAD_COND_INITIALIZER;
pthread_cond_t conn_cv = PTHREAD_COND_INITIALIZER;
pthread_cond_t work_full = PTHREAD_COND_INITIALIZER;
pthread_cond_t work_empty = PTHREAD_COND_INITIALIZER;

int idle_thread = 4, acceptor_num = 4, worker_num = 16, work_queue_len = 256;
int conn_front = -1, conn_rear = -1;
int work_front = -1, work_rear = -1;
int *conn_queue;
work *work_queue;

void
serve_connection (int sockfd);

void idle_acceptor_wait();
void idle_acceptor_signal();
int conn_wait();
void conn_signal(int sockfd);
work work_get();
void work_push(connection_t *conn, int number);
char is_Prime(int number);

void*
acceptorThread(void *arg)
{
  int sockfd;
  while(!shutting_down)
  {
    sockfd = conn_wait();
    if(shutting_down)
      break;
    serve_connection(sockfd);
    if(shutting_down)
      break;
    idle_acceptor_signal();
  }
  pthread_exit(NULL);
}

void*
workerThread(void *arg)
{
  work temp;
  ssize_t result, n;
  char response[MAXLINE];
  while(!shutting_down)
  {
    temp = work_get();
    if(shutting_down)
      break;
    if(is_Prime(temp.number))
      sprintf(response, "%d is prime number\n", temp.number);
    else
      sprintf(response, "%d isn't prime number\n", temp.number);
    n = (ssize_t)strlen(response);
    result = writen(temp.conn, response, n);
    if (result != n) {
      perror ("written failed");
      CHECK (close (temp.conn->sockfd));
    }
  }
  pthread_exit(NULL);
}

void
idle_acceptor_wait(){
  pthread_mutex_lock(&conn_mutex);
  while(idle_thread == 0 && !shutting_down) pthread_cond_wait(&idle_acceptor, &conn_mutex);
  idle_thread--;
  pthread_mutex_unlock(&conn_mutex);
}

void
idle_acceptor_signal(){
  pthread_mutex_lock(&conn_mutex);
  idle_thread++;
  if(idle_thread == 1) {
    pthread_mutex_unlock(&conn_mutex);
    pthread_cond_signal(&idle_acceptor);
  }
  else
    pthread_mutex_unlock(&conn_mutex);
}

int
conn_wait(){
  int sockfd;
  pthread_mutex_lock(&conn_mutex);
  while(conn_front == conn_rear && !shutting_down) pthread_cond_wait(&conn_cv, &conn_mutex);
  if(shutting_down){
    pthread_mutex_unlock(&conn_mutex);
    return 0;
  }
  conn_rear = (conn_rear + 1) % acceptor_num;
  sockfd = conn_queue[conn_rear];
  pthread_mutex_unlock(&conn_mutex);
  return sockfd;
}

void
conn_signal(int sockfd){
  pthread_mutex_lock(&conn_mutex); 
  conn_front = (conn_front + 1) % acceptor_num;
  conn_queue[conn_front] = sockfd;
  if((conn_rear + 1) % acceptor_num == conn_front) {
    pthread_mutex_unlock(&conn_mutex);
    pthread_cond_signal(&conn_cv);
  }
  else
    pthread_mutex_unlock(&conn_mutex);
}

void
work_push(connection_t *conn, int number){
  work new_work;
  new_work.conn = conn;
  new_work.number = number;
  pthread_mutex_lock(&work_mutex);
  while((work_front + 1) % work_queue_len == work_rear && !shutting_down) pthread_cond_wait(&work_empty, &work_mutex);
  work_front = (work_front + 1) % work_queue_len;
  work_queue[work_front] = new_work;
  if((work_rear + 1) % work_queue_len == work_front && !shutting_down) {
    pthread_mutex_unlock(&work_mutex);
    pthread_cond_signal(&work_full);
  }
  else
    pthread_mutex_unlock(&work_mutex);
}

work
work_get(){
  work temp;
  pthread_mutex_lock(&work_mutex);
  while(work_front == work_rear && !shutting_down) pthread_cond_wait(&work_full, &work_mutex);
  if(shutting_down){
    pthread_mutex_unlock(&work_mutex);
    temp.conn = NULL;
    temp.number = 0;
    return temp;
  }
  work_rear = (work_rear + 1) % work_queue_len;
  temp = work_queue[work_rear];
  if((work_front + 2) % work_queue_len == work_rear){
    pthread_mutex_unlock(&work_mutex);
    pthread_cond_signal(&work_empty);
  }
  else
    pthread_mutex_unlock(&work_mutex);
  return temp;
}

void
server_handoff (int sockfd) {
  conn_signal(sockfd);
}

/* the main per-connection service loop of the server; assumes
   sockfd is a connected socket */

char
is_Prime(int num)
{
  if(num % 2 == 0)
    return FALSE;
  for(int i = 3; i * i <= num; i+=2)
    if(num % i ==0) return FALSE;
  return TRUE;
}

void
serve_connection (int sockfd) {
  ssize_t  n/*, result*/;
  char line[MAXLINE];
  connection_t conn;
  connection_init (&conn);
  conn.sockfd = sockfd;
  int number;
  while (! shutting_down) {
    if ((n = readline (&conn, line, MAXLINE)) == 0) goto quit;
    /* connection closed by other end */
    if (shutting_down) goto quit;
    if (n < 0) {
      perror ("readline failed");
      goto quit;
    }
    number = atoi(line);
    work_push(&conn, number);
    /*if(is_Prime(number))
      sprintf(line, "%d is prime number\n", number);
    else
      sprintf(line, "%d isn't prime number\n", number);
    n = (ssize_t)strlen(line);
    result = writen (&conn, line, n);
    if (shutting_down) goto quit;
    if (result != n) {
      perror ("writen failed");
      goto quit;
    }*/
  }
quit:
  CHECK (close (conn.sockfd));
}

/* set up socket to use in listening for connections */
void
open_listening_socket (int *listenfd) {
  struct sockaddr_in servaddr;
  const int server_port = 0; /* use ephemeral port number */
  socklen_t namelen;
  memset (&servaddr, 0, sizeof(struct sockaddr_in));
  servaddr.sin_family = AF_INET;
  /* htons translates host byte order to network byte order; ntohs
     translates network byte order to host byte order */
  servaddr.sin_addr.s_addr = htonl (INADDR_ANY);
  servaddr.sin_port = htons (server_port);
  /* create the socket */
  CHECK (*listenfd = socket(AF_INET, SOCK_STREAM, 0))
  /* bind it to the ephemeral port number */
  CHECK (bind (*listenfd, (struct sockaddr *) &servaddr, sizeof (servaddr)));
  /* extract the ephemeral port number, and put it out */
  namelen = sizeof (servaddr);
  CHECK (getsockname (*listenfd, (struct sockaddr *) &servaddr, &namelen));
  fprintf (stderr, "server using port %d\n", ntohs(servaddr.sin_port));
}

/* handler for SIGINT, the signal conventionally generated by the
   control-C key at a Unix console, to allow us to shut down
   gently rather than having the entire process killed abruptly. */ 
void
siginthandler (int sig, siginfo_t *info, void *ignored) {
  shutting_down = 1;
  pthread_cond_broadcast(&work_full);
  pthread_cond_broadcast(&work_empty);
  pthread_cond_broadcast(&conn_cv);
  pthread_cond_signal(&idle_acceptor);
}

void
install_siginthandler () {
  struct sigaction act;
  /* get current action for SIGINT */
  CHECK (sigaction (SIGINT, NULL, &act));
  /* add our handler */
  act.sa_sigaction = siginthandler;
  /* update action for SIGINT */
  CHECK (sigaction (SIGINT, &act, NULL));
}

int
main (int argc, char **argv) {
  int connfd, listenfd;
  socklen_t clilen;
  struct sockaddr_in cliaddr;
  int opt;
  pthread_t *acceptorThreads, *workerThreads;
  while((opt=getopt(argc,argv,"a:w:")) != -1)
  {
    if(opt == 'a')
      acceptor_num = atoi(optarg);
    else if(opt == 'w'){
      worker_num = atoi(optarg);
      work_queue_len = worker_num * 256;
    }
  }
  idle_thread = acceptor_num;
  conn_queue = (int*)malloc(acceptor_num*sizeof(int));
  work_queue = (work*)malloc(work_queue_len*sizeof(work));
  acceptorThreads = (pthread_t*)malloc(acceptor_num*sizeof(pthread_t));
  workerThreads = (pthread_t*)malloc(worker_num*sizeof(pthread_t));
  for(int i = 0; i < acceptor_num; i++)
    pthread_create(&acceptorThreads[i], NULL, acceptorThread, NULL);
  for(int i = 0; i < worker_num; i++)
    pthread_create(&workerThreads[i], NULL, workerThread, NULL);

  /* NOTE: To make this multi-threaded, You may need insert
     additional initialization code here, but you will not need to
     modify anything below here, though you are permitted to
     change anything in this file if you feel it is necessary for
     your design */


  install_siginthandler();
  open_listening_socket (&listenfd);
  CHECK (listen (listenfd, 4));
  /* allow up to 4 queued connection requests before refusing */
  while (! shutting_down) {
    idle_acceptor_wait();
    errno = 0;
    clilen = sizeof (cliaddr); /* length of address can vary, by protocol */
    if ((connfd = accept (listenfd, (struct sockaddr *) &cliaddr, &clilen)) < 0) {
      if (errno != EINTR) ERR_QUIT ("accept"); 
      /* otherwise try again, unless we are shutting down */
    } else {
     server_handoff (connfd); /* process the connection */
    }
  }
  CHECK (close (listenfd));
  for(int i = 0; i < acceptor_num; i++)
    pthread_join(acceptorThreads[i], NULL);
  for(int i = 0; i < worker_num; i++)
    pthread_join(workerThreads[i], NULL);
  pthread_mutex_destroy(&conn_mutex);
  pthread_mutex_destroy(&work_mutex);
  pthread_cond_destroy(&idle_acceptor);
  pthread_cond_destroy(&conn_cv);
  pthread_cond_destroy(&work_full);
  pthread_cond_destroy(&work_empty);
  free(acceptorThreads);
  free(workerThreads);
  free(conn_queue);
  free(work_queue);
  return 0;
}
