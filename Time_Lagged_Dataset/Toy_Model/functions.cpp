#include "create_time_lagged_dataset.h"

using namespace std;
using namespace arma;

// #! FIELDS time deep.node-0 deep.node-1 p.x p.y ene.bias opes.bias
void read_input(string filename) {

	ifstream input;
	input.open(filename);

	string par1,par2,par3;

	cout << "#--------- parameters ---------#" << endl;
	while ( input >> par1 >> par2 >> par3)
	{
		if (par1 == "colvar")
		{
			colvar = par3;
			cout << "colvar file " << colvar << endl;
		}	
		else if (par1 == "temp")
		{
			temp = stof(par3);
			cout << "temperature " << temp << endl;
		}
		else if (par1 == "lag")
		{
			lag = stof(par3);
			cout << "lag time " << lag << endl;
		}
		else if (par1 == "first_index")
		{
			first_index = stoi(par3);
		}
		else if (par1 == "last_index")
		{
			last_index = stoi(par3);
			cout << "indeces for descriptors [" << first_index << "," << last_index<<"]\n";
		}
		else if (par1 == "if_weights")
		{
			if_weights = stoi(par3);
		}
		else if (par1 == "weights_index")
		{
			weight_index = stoi(par3);
			cout << "index for weights "<< weight_index << "\n";
		}
		else
		{
			cout << "parameter " << par1 << " it is not acceptable, ignore" << endl;
		}
	}
	
	cout << "#--------- end parameters -----#" << endl;
	input.close();
}


template<typename type>
type log_sum_exp(Mat<type> v) {
	return as_scalar( log( sum( exp(v) ) ) );
}

template<typename type>
int closest_idx(Mat<type> v, type value) {

	if ( not v.is_sorted() )
		v = sort(v);

	auto idx = lower_bound(v.begin(),v.end(),value);
	return (idx-v.begin())-1;

}

template<typename type>
void find_time_lagged_configurations(vector<fstream*> file, Mat<type> x,Mat<type> t, mytype lag) {
 
	vector<Mat<type>> data;

	uword N = t.size();
	//find maximum time idx
	uword idx_end = closest_idx(t,t(N-1)-lag);
	mytype stop_condition,deltaTau;
	uword n_j;
	//counter for for loops
	uword i,j,k;
	uword cols = x.n_cols;
	
	cout << "idx end " << idx_end << endl;
	//loop over time array and find pairs which are far away by lag
	for (i=0; i<idx_end;i++)
	{
		stop_condition = lag+t(i+1);
		n_j=0;
		for (j=i;j<N;j++) 
		{
			if ( t(j)<stop_condition && t(j+1)>t(i)+lag )
			{
				//save on file the found couples
				for (k=0; k<cols-1 ; k++)
				{
					//x_t
					*file[0] << x.row(i)(k) << "\t";
					//x_lag
					*file[1] << x.row(j)(k) << "\t";
				}
				*file[0] << x.row(i)(cols-1) << "\n";
				*file[1] << x.row(j)(cols-1) << "\n";
				
				//w_lag
				deltaTau = min( t(i+1)+lag , t(j+1)) - max(t(i)+lag, t(j));
				*file[2] << deltaTau << "\n";
				n_j++;
			}
			else if (t(j)>stop_condition)
				break;
		}
		for (k=0;k<n_j;k++) 
		{
			//w_t
			*file[3] << (t(i+1)-t(i)) / float(n_j) << "\n";
		}
	}
}

template<typename type>
void create_time_lagged_dataset(vector<fstream*> file, Mat<type> X, Mat<type> t, Mat<type> logweights, mytype lag_time) {

	mytype dt,lognorm;
	Mat<type> d_tprime,tprime;
	
	if (t.is_empty())
		t = linspace<Mat<type>>(0, t.n_elem, t.n_elem);

	if (logweights.is_empty()) 
	{
		cout << "logweight None" << endl;
		tprime=t;
	}
	else
	{
		//logw = beta*V
		logweights/=temp;
		//time increment in simulation
		dt = t(1)-t(0);
		dt = round( dt * 1000.0 ) / 1000.0; // 3 decimal places	
		//sanitize logweights
		logweights-=logweights.max();
		lognorm = log_sum_exp(logweights);
		logweights/=lognorm;
		//compute istantaneous time increment in rescaled time t'
		d_tprime = exp(logweights) * dt;
		//calculate cumulative time t'
		tprime = cumsum(d_tprime);
	}
	find_time_lagged_configurations(file,X,tprime,lag_time);
}
