#ifndef _TINN_H_
#define _TINN_H_

typedef struct
{
    double* w;
    double* b;
    double* h;
    double* o;
    int nb;
    int nips;
    int nhid;
    int nops;
}
Tinn;

extern double xttrain(Tinn, double* in, double* tg, double rate);

extern Tinn xtbuild(int nips, int nhid, int nops);

extern void xtfree(Tinn);

#endif
