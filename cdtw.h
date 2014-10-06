#include <stdlib.h>

typedef struct Path
{
  int k;
  int *px;
  int *py;
} Path;

int path(double *cost, int n, int m, int startx, int starty, Path *p);
double min3(double a, double b, double c);