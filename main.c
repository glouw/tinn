#include "Tinn.h"

#include <stdio.h>
#include <stdlib.h>

static double* inload(int nips)
{
    double* in = (double*) calloc(nips, sizeof(*in));
    in[0] = 0.05;
    in[1] = 0.10;
    return in;
}

static double* tgload(int nops)
{
    double* tg = (double*) calloc(nops, sizeof(*tg));
    tg[0] = 0.01;
    /* tg[1] = 0.99; */
    return tg;
}

int main()
{
    int nips = 2;
    int nhid = 3;
    int nops = 1;
    double* in = inload(nips);
    double* tg = tgload(nops);
    Tinn tinn = tbuild(nips, nops, nhid);
    int i;
    for(i = 0; i <= 10000; i++)
        printf("%.18f\n", ttrain(tinn, in, tg, 0.5));
    tfree(tinn);
    free(in);
    free(tg);
    return 0;
}
