FLAGS = -std=c++14 -DARMA_DONT_USE_WRAPPER -larmadillo -llapack -lblas -fopenmp -O3 -lpthread 
arma_dir=/home/npedrani@iit.local/programs/armadillo-11.0.1/install
LIB=-I/$(arma_dir)/include -L/$(arma_dir)/lib

compila: create_time_lagged_dataset.cpp functions.cpp create_time_lagged_dataset.h
	g++ $(LIB) create_time_lagged_dataset.cpp -o create_time_lagged_dataset.exe $(FLAGS)

clean:  
	rm create_time_lagged_dataset.exe
 
