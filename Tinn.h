#ifndef _TINN_H_
#define _TINN_H_

/*
 * TINN - The tiny dependency free ANSI-C feed forward neural network
 * library with one hidden layer back propogation support.
 */

typedef struct
{
    double* o;
    double* h;
    double* w;
    int nops;
    int nhid;
    int nips;
}
Tinn;

double ttrain(Tinn, double* in, double* tg, double rate);

Tinn tbuild(int inputs, int output, int hidden);

void tfree(Tinn);

#endif
