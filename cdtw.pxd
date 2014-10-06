cdef extern from "cdtw.h":
	ctypedef struct Path:
		int k
		int *px
		int *py

	int path(double *cost, int n, int m, int startx, int starty, Path *p)
	double min3(double a, double b, double c)