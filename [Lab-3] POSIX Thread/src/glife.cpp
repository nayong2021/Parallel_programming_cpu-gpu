#include "glife.h"
using namespace std;

int gameOfLife(int argc, char *argv[]);
void singleThread(int, int);
void* workerThread(void *);
int nprocs, display;
pthread_barrier_t barrier;
GameOfLifeGrid* g_GameOfLifeGrid;
struct thread_argument {
  int row_from, row_to, cols;
};

uint64_t dtime_usec(uint64_t start)
{
  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

GameOfLifeGrid::GameOfLifeGrid(int rows, int cols, int gen)
{
  m_Generations = gen;
  m_Rows = rows;
  m_Cols = cols;

  m_Grid = (int**)malloc(sizeof(int*) * rows);
  if (m_Grid == NULL) 
    cout << "1 Memory allocation error " << endl;

  m_Temp = (int**)malloc(sizeof(int*) * rows);
  if (m_Temp == NULL) 
    cout << "2 Memory allocation error " << endl;

  m_Grid[0] = (int*)malloc(sizeof(int) * (cols*rows));
  if (m_Grid[0] == NULL) 
    cout << "3 Memory allocation error " << endl;

  m_Temp[0] = (int*)malloc(sizeof(int) * (cols*rows));	
  if (m_Temp[0] == NULL) 
    cout << "4 Memory allocation error " << endl;

  for (int i = 1; i < rows; i++) {
    m_Grid[i] = m_Grid[i-1] + cols;
    m_Temp[i] = m_Temp[i-1] + cols;
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      m_Grid[i][j] = m_Temp[i][j] = 0;
    }
  }
}

// Entry point
int main(int argc, char* argv[])
{
  if (argc != 7) {
    cout <<"Usage: " << argv[0] << " <input file> <display> <nprocs>"
           " <# of generation> <width> <height>" << endl;
    cout <<"\n\tnprocs = 0: Running sequentiallyU" << endl;
    cout <<"\tnprocs =1: Running on a single thread" << endl;
    cout <<"\tnprocs >1: Running on multiple threads" << endl;
    cout <<"\tdisplay = 1: Dump results" << endl;
    return 1;
  }

  return gameOfLife(argc, argv);
}

int gameOfLife(int argc, char* argv[])
{
  int cols, rows, gen;
  ifstream inputFile;
  int input_row, input_col;
  uint64_t difft;
  pthread_t *threadID;

  inputFile.open(argv[1], ifstream::in);

  if (inputFile.is_open() == false) {
    cout << "The "<< argv[1] << " file can not be opend" << endl;
    return 1;
  }

  display = atoi(argv[2]);
  nprocs = atoi(argv[3]);
  gen = atoi(argv[4]);
  cols = atoi(argv[5]);
  rows = atoi(argv[6]);

  g_GameOfLifeGrid = new GameOfLifeGrid(rows, cols, gen);

  while (inputFile.good()) {
    inputFile >> input_row >> input_col;
    if (input_row >= rows || input_col >= cols) {
      cout << "Invalid grid number" << endl;
      return 1;
    } else
      g_GameOfLifeGrid->setCell(input_row, input_col);
  }

  // Start measuring execution time
  difft = dtime_usec(0);

  // TODO: YOU NEED TO IMPLMENT THE SINGLE THREAD and PTHREAD
  if (nprocs == 0) {
    // Running with your sequential version
    while(g_GameOfLifeGrid->getGens()){
      for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
	  if(g_GameOfLifeGrid->isLive(i, j)){
	    int NumOfNeighbors = g_GameOfLifeGrid->getNumOfNeighbors(i,j);
	    if(NumOfNeighbors == 2 || NumOfNeighbors == 3)
	      g_GameOfLifeGrid->live(i,j);
	    else
	      g_GameOfLifeGrid->dead(i,j);
	  }
          else{
	    if(g_GameOfLifeGrid->getNumOfNeighbors(i,j) == 3)
	      g_GameOfLifeGrid->live(i,j);
	    else
	      g_GameOfLifeGrid->dead(i,j);
	  }
        }
      }
      g_GameOfLifeGrid->next();
      g_GameOfLifeGrid->decGen();
    }
  } else { 
    // Running multiple threads (pthread)
    int rc = pthread_barrier_init(&barrier, NULL, (unsigned int)nprocs);
    if(rc){
      printf("Error; return code from pthread_barrir_init() is %d\n", rc);
      exit(-1);
    }
    pthread_t* threads;
    threads = (pthread_t*)malloc(sizeof(pthread_t)*nprocs);
    for(int i = 0; i < nprocs; i++){
      thread_argument* arg;
      arg = (thread_argument*)malloc(sizeof(thread_argument));
      arg->cols = cols;
      arg->row_from = i * (rows / nprocs);
      if(i == nprocs-1)
        arg->row_to = rows;
      else
	arg->row_to = arg->row_from + (rows / nprocs);
      rc = pthread_create(&threads[i], NULL, workerThread, (void*)arg);
      if(rc){
        printf("Error; return code from pthread_create() is %d\n", rc);
        exit(-1);
      }
    }
    for(int i = 0; i < nprocs; i++){
      rc = pthread_join(threads[i], NULL);
      if(rc){
        printf("Error; return code from pthread_join() is %d\n", rc);
        exit(-1);
      }
    }
  }

  difft = dtime_usec(difft);

  // Print indices only for running on CPU(host).
  if(display){
    g_GameOfLifeGrid->dump();
    g_GameOfLifeGrid->dumpIndex();
  }
  if (nprocs == 0) {
    // Sequential version
    cout << "Execution time(seconds) on Serial version: " << difft/(float)USECPSEC << endl;
  } else if (nprocs >= 1) {
    // Single or multi-thread execution time 
    cout << "Execution time(seconds) on thread version: " << difft/(float)USECPSEC << endl;
  }
  inputFile.close();
  cout << "Program end... " << endl;
  return 0;
}

// TODO: YOU NEED TO IMPLMENT PTHREAD
void* workerThread(void *argument)
{
  int rc = 0;
  thread_argument* arg = (thread_argument*)argument;
  while(g_GameOfLifeGrid->getGens()){
    for(int i = arg->row_from; i < arg->row_to; i++){
      for(int j = 0; j < arg->cols; j++){
        if(g_GameOfLifeGrid->isLive(i, j)){
	  int NumOfNeighbors = g_GameOfLifeGrid->getNumOfNeighbors(i,j);
	  if(NumOfNeighbors == 2 || NumOfNeighbors == 3)
	    g_GameOfLifeGrid->live(i,j);
	  else
	    g_GameOfLifeGrid->dead(i,j);
        }
	else{
	  if(g_GameOfLifeGrid->getNumOfNeighbors(i,j) == 3)
	    g_GameOfLifeGrid->live(i,j);
	  else
	    g_GameOfLifeGrid->dead(i,j);
	}
      }
    }
    int rc = pthread_barrier_wait(&barrier); 
    if(rc != 0){
      if(rc == PTHREAD_BARRIER_SERIAL_THREAD)
         g_GameOfLifeGrid->decGen();
      else {
        printf("Error; return code from pthread_barrier_wait() is %d\n", rc);
        exit(-1);
      }
    }
    g_GameOfLifeGrid->next(arg->row_from, arg->row_to);
    rc = pthread_barrier_wait(&barrier); 
    if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD){
      printf("Error; return code from pthread_barrier_wait() is %d\n", rc);
      exit(-1);
    }
  }
}

// HINT: YOU MAY NEED TO FILL OUT BELOW FUNCTIONS OR CREATE NEW FUNCTIONS
// HINT: YOU MAY NEED TO FILL OUT BELOW FUNCTIONS OR CREATE NEW FUNCTIONS
void GameOfLifeGrid::next(const int from, const int to)
{
  for(int i = from; i < to; i++)
    for(int j = 0; j < m_Cols; j++)
      m_Grid[i][j] = m_Temp[i][j];
}

void GameOfLifeGrid::next()
{
  for(int i = 0; i < m_Rows; i++)
    for(int j = 0; j < m_Cols; j++)
      m_Grid[i][j] = m_Temp[i][j];
}

// TODO: YOU MAY NEED TO IMPLMENT IT TO GET NUMBER OF NEIGHBORS 
int GameOfLifeGrid::getNumOfNeighbors(int rows, int cols)
{
  int numOfNeighbors = 0;
  if(rows != 0 && cols != 0)
    numOfNeighbors += m_Grid[rows-1][cols-1];
  if(rows != 0)
    numOfNeighbors += m_Grid[rows-1][cols];
  if(cols != 0)
    numOfNeighbors += m_Grid[rows][cols-1];
  if(rows != 0 && cols != m_Cols-1)
    numOfNeighbors += m_Grid[rows-1][cols+1];
  if(cols != m_Cols-1)
    numOfNeighbors += m_Grid[rows][cols+1];
  if(rows != m_Rows-1 && cols != 0)
    numOfNeighbors += m_Grid[rows+1][cols-1];
  if(rows != m_Rows-1)
    numOfNeighbors += m_Grid[rows+1][cols];
  if(rows != m_Rows-1 && cols != m_Cols-1)
    numOfNeighbors += m_Grid[rows+1][cols+1];
  return numOfNeighbors;
}

void GameOfLifeGrid::dump() 
{
  cout << "===============================" << endl;

  for (int i = 0; i < m_Rows; i++) {
    cout << "[" << i << "] ";
    for (int j = 0; j < m_Cols; j++) {
      if (m_Grid[i][j] == 1)
        cout << "*";
      else
        cout << "o";
    }
    cout << endl;
  }
  cout << "===============================\n" << endl;
}

void GameOfLifeGrid::dumpIndex()
{
  cout << ":: Dump Row Column indices" << endl;
  for (int i=0; i < m_Rows; i++) {
    for (int j=0; j < m_Cols; j++) {
      if (m_Grid[i][j]) cout << i << " " << j << endl;
    }
  }
}
