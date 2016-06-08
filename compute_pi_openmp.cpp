
#include <stdio.h>

double f(double y) { return(4.0/(1.0 + y*y)); }

int main() {

  double w, x, sum, pi;
  int i;
  int n = 1000000;
  w = 1.0/n;
  sum = 0.0;

  #pragma omp parallel for private(x) shared(w) reduction(+:sum)
    for(i = 0; i < n; i++)
    {
      x = w*(i - 0.5);
      sum = sum + f(x);
    }
    pi = w*sum;
    printf("pi = %f\n", pi);

  return 0;
}