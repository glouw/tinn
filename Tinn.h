#ifndef _TINN_H_
#define _TINN_H_

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

double xttrain(Tinn, double* in, double* tg, double rate);

Tinn xtbuild(int nips, int nops, int nhid);

void xtfree(Tinn);

#endif
