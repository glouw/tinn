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

extern double xttrain(Tinn, double* in, double* tg, double rate);

extern Tinn xtbuild(int nips, int nops, int nhid);

extern void xtfree(Tinn);

#endif
