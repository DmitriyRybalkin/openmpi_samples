
#include <stdio.h>
#include <omp.h>

#define N 4096

double a[N][N], b[N][N], c[N][N];

int main() {
  int i, j, k;
  double t1, t2;

  /* matrix initialization */
  printf("Matrix initialization\n");
  for(i = 0; i < N; i++)
    for(j = 0; j < N; j++)
      a[i][j] = b[i][j] = i*j;

  t1 = omp_get_wtime();
  /* main computational block */
  printf("Matrix multiplication\n");
  #pragma omp parallel for shared(a, b, c) private(i, j, k)
    for(i = 0; i < N; i++) {
      for(j = 0; j < N; j++) {
	c[i][j] = 0.0;
	for(k = 0; k < N; k++) c[i][j] += a[i][k]*b[k][j];
      }
    }
  t2 = omp_get_wtime();
  printf("Time = %lf\n", t2 - t1);

  return 0;
}