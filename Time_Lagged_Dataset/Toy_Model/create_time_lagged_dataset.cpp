#include "functions.cpp"

using namespace std;
using namespace arma;

//osservazione: posso migliorare la performance di scrittura e lettura di file:
// prima di tutto facendo in binario, successivamente usando write() al posto di fstream

int main(int argc, char **argv)
{
  //measure time for efficiency
  wall_clock timer;
  timer.tic();

  /* test for readlines from file */
  if(argc != 2)
  {
	  cout << "please insert input.dat" << endl;
	  return -1;
  }

  // read input parameters
	read_input(argv[1]);
  // read COLVAR file
  ifstream COLVAR; COLVAR.open(colvar);
  // skip first line
  string str;
  getline(COLVAR, str);
  cout << "HEADER of data file" << endl;
  cout << str << endl;
  //write directly on file
  /* 4 files:
		xt
		xlag
		wlag
		wt */ 
  fstream xt("x_t.txt",ios::app),xlag("x_lag.txt",ios::app),wlag("w_lag.txt",ios::app),wt("w_t.txt",ios::app);
  vector<fstream*> file;
  file.push_back(&xt);file.push_back(&xlag);file.push_back(&wlag);file.push_back(&wt);
  
  /* load data from file*/
  Mat<mytype> data;
  data.load(COLVAR);
  //close data file
  COLVAR.close();
  
  /* X: descriptors; t: time; logweights: V */
  Mat<mytype> X, t, logweights;
  X = data.cols( first_index, last_index );
  t = data.col( 0 );

  if (if_weights){
	  logweights = data.col(weight_index);
	  cout << "with weights" << endl;
  }

  /* create the time lagged data set */
  create_time_lagged_dataset(file ,X, t, logweights, lag);

  //close files
  for (auto el : file)
	(*el).close();

  mytype time = timer.toc();
  cout << "time: " << time << "s" << endl;
  return 0;
}