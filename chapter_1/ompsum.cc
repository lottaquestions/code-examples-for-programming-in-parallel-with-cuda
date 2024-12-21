#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <numbers> 
#include "cxtimers.h"

// To compile: /home/nick/Software/gcc13.2/installation/bin/g++-13.2 -std=c++20 -fopenmp ompsum.cc -o ompsum.bin

// To run: LD_LIBRARY_PATH=/home/nick/Software/gcc13.2/installation/lib64 ./ompsum.bin

inline float sinsum(float x, int terms){
    // sin(x) = x - x^3/3! + x^5/5!...
    float term = x; // First term of series
    float sum = term; // Sum of terms so far
    float x2 = x*x;

    for(int n = 1; n < terms; n++){
        term *= -x2 /(float) (2*n*(2*n+1));
        sum += term;
    }
    return sum;

}

int main(int argc, char *argv[]){
    int steps = (argc > 1) ? atoi (argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi (argv[2]) : 1000;
    int threads = (argc > 3) ? atoi(argv[3]) : 4;

    double step_size = std::numbers::pi / (steps -1); // n-1 steps

    cx::timer tim;
    double omp_sum = 0.0;
    omp_set_num_threads(threads);
    #pragma omp parallel for reduction (+:omp_sum)
    for (int step = 0; step < steps; step++){
        float x = step_size * step;
        omp_sum +=sinsum(x, terms); // Sum of Taylor series
    }
    double cpu_time = tim.lap_ms(); // Elapsed time

    // Trapezoidal Rule correction
    omp_sum -= 0.5 * (sinsum(0.0, terms) + sinsum (std::numbers::pi, terms));
    omp_sum *= step_size;
    std::cout << "omp_sum = " << omp_sum << ", steps " << steps << " terms " << terms << " time " << cpu_time << " threads "<< threads << std::endl;
}