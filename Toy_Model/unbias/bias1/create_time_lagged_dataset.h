#include <iostream>
#include <armadillo>
#include <vector>
#include <cmath>
#include <algorithm> // std::lower_bound, std::sort
#include <fstream>

using namespace std;
using namespace arma;

typedef float mytype;

// input parameters
mytype beta, lag;
string colvar;
uword first_index,last_index, skip_rows;
bool if_weights;
uword weight_index;

void read_input(string);

template<typename type>
type log_sum_exp(Mat<type>); 

template<typename type>
int closest_idx(Mat<type>,type);

template<typename type>
void find_time_lagged_configurations(vector<fstream*>, Mat<type>, Mat<type>, mytype);

template<typename type>
void create_time_lagged_dataset(vector<fstream*> ,Mat<type>, Mat<type> t = {}, Mat<type> logweights = {}, mytype lag_time=10);