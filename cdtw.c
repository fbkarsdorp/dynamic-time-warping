#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cdtw.h"

double min3(double a, double b, double c) {
	double min;
    min = a;
    if (b < min)
        min = b;
    if (c < min)
        min = c;
    return min;
}

// Compute the warp path starting at cost[startx, starty]
// If startx = -1 -> startx = n-1; if starty = -1 -> starty = m-1
int
path(double *cost, int n, int m, int startx, int starty, Path *p)
{
  int i, j, k, z1, z2;
  int *px;
  int *py;
  double min_cost;
  
  if ((startx >= n) || (starty >= m))
    return 0;
  
  if (startx < 0)
    startx = n - 1;
  
  if (starty < 0)
    starty = m - 1;
      
  i = startx;
  j = starty;
  k = 1;
  
  // allocate path for the worst case
  px = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));
  py = (int *) malloc ((startx+1) * (starty+1) * sizeof(int));
  
  px[0] = i;
  py[0] = j;
  
  while ((i > 0) || (j > 0))
    {
      if (i == 0)
  j--;
      else if (j == 0)
  i--;
      else
  {
    min_cost = min3(cost[(i-1)*m+j],
        cost[(i-1)*m+(j-1)], 
        cost[i*m+(j-1)]);
    
    if (cost[(i-1)*m+(j-1)] == min_cost)
      {
        i--;
        j--;
      }
    else if (cost[i*m+(j-1)] == min_cost)
      j--;
    else
      i--;
  }
      
      px[k] = i;
      py[k] = j;
      k++;      
    }
  
  p->px = (int *) malloc (k * sizeof(int));
  p->py = (int *) malloc (k * sizeof(int));
  for (z1=0, z2=k-1; z1<k; z1++, z2--)
    {
      p->px[z1] = px[z2];
      p->py[z1] = py[z2];
    }
  p->k = k;
  
  free(px);
  free(py);
  
  return 1;
}