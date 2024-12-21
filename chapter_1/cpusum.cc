#include <iostream>
#include <stdlib.h>
#include <numbers> 
#include "cxtimers.h"

// To compile: /home/nick/Software/gcc13.2/installation/bin/g++-13.2 -std=c++20 cpusum.cc -o cpusum.bin

// To run: LD_LIBRARY_PATH=/home/nick/Software/gcc13.2/installation/lib64 ./cpusum.bin

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

    double step_size = std::numbers::pi / (steps -1); // n-1 steps

    cx::timer tim;
    double cpu_sum = 0.0;
    for (int step = 0; step < steps; step++){
        float x = step_size * step;
        cpu_sum +=sinsum(x, terms); // Sum of Taylor series
    }
    double cpu_time = tim.lap_ms(); // Elapsed time

    // Trapezoidal Rule correction
    cpu_sum -= 0.5 * (sinsum(0.0, terms) + sinsum (std::numbers::pi, terms));
    cpu_sum *= step_size;
    std::cout << "cpu_sum = " << cpu_sum << ", steps " << steps << " terms " << terms << " time " << cpu_time << std::endl;
}