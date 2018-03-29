#ifndef _TINN_H_
#define _TINN_H_

/*
 * TINN - The tiny dependency free ANSI-C feed forward neural network
 * library with one hidden layer back propogation support.
 */

typedef struct
{
    double* O;
    double* H;
    double* W;
    int output;
    int hidden;
    int inputs;
}
Tinn;

double ttrain(Tinn, double* I, double* T, double rate);

Tinn tnew(int inputs, int output, int hidden);

void tfree(Tinn);

#endif
