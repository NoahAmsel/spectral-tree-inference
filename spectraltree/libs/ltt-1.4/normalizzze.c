#include "mex.h"
#include <string.h>
#include <stdio.h>

void normalize_cols(double *y, int rows, int cols)
{
  int i, j;
  double *z, sum;
  for (i=0; i<cols; i++) {
    /* calculate column sum */
    z = y;
    sum = 0;
    for (j=0; j<rows; j++) {
      if (*z < 0) 
          mexErrMsgTxt("[normalizzze.c] One inputs must be non-negative.");
      sum += *z;
      z++;
    }
    /* divide by the sum */
    /*printf("sum %g \n", sum);*/
    if (sum > 0) {
      z = y;
      for (j=0; j<rows; j++) {
        /*printf("%g --> ", *z);*/
     	*z = (*z)/sum;
        /*printf("%g \n", *z);*/
        z++;
      }
    }
    /* next row */
    y = z;
  }
}

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  double    *x, *y;
  int       rows, cols, status, ndims, i;
  const int *dims;

  /*  check for proper number of arguments */
  if(nrhs!=1) 
    mexErrMsgTxt("[normalizzze.c] One input required.");
  if(nlhs>1) 
    mexErrMsgTxt("[normalizzze.c] Too many output arguments.");
  
  /*  create a pointer to the input matrix x */
  x = mxGetPr(prhs[0]);
  
  /*  get the dimensions of the matrix input x */
  rows  = mxGetM(prhs[0]);
  cols  = mxGetN(prhs[0]);
  ndims = mxGetNumberOfDimensions(prhs[0]);
  dims  = mxGetDimensions(prhs[0]);

  /* the output pointer to the output matrix */
  plhs[0] = mxCreateDoubleMatrix(rows, cols, mxREAL);
  
  /*  create a C pointer to a copy of the output matrix */
  y = mxGetPr(plhs[0]);
  memcpy(y, x, sizeof(double)*rows*cols);
  
  /*  call the C subroutine */
  normalize_cols(y, rows, cols);
  
  /*  reshape the output */
  if (mxSetDimensions(plhs[0], dims, ndims))
    mexErrMsgTxt("[normalizzze.c] Reshape failed");
}
