#include "Tinn.h"

#include <stdlib.h>
#include <math.h>
#include <time.h>

static double error(Tinn t, double* tg)
{
    double error = 0.0;
    int i;
    for(i = 0; i < t.nops; i++)
        error += 0.5 * pow(tg[i] - t.o[i], 2.0);
    return error;
}

static void backwards(Tinn t, double* in, double* tg, double rate)
{
    double* x = t.w + t.nhid * t.nips;
    int i;
    for(i = 0; i < t.nhid; i++)
    {
        double sum = 0.0;
        int j;
        for(j = 0; j < t.nops; j++)
        {
            double a = t.o[j] - tg[j];
            double b = t.o[j] * (1 - t.o[j]);
            double c = x[j * t.nhid + i];
            sum += a * b * c;
        }
        for(j = 0; j < t.nips; j++)
        {
            double a = sum;
            double b = t.h[i] * (1 - t.h[i]);
            double c = in[j];
            t.w[i * t.nips + j] -= rate * a * b * c;
        }
        for(j = 0; j < t.nops; j++)
        {
            double a = t.o[j] - tg[j];
            double b = t.o[j] * (1 - t.o[j]);
            double c = t.h[i];
            x[j * t.nhid + i] -= rate * a * b * c;
        }
    }
}

static double act(double net)
{
    return 1.0 / (1.0 + exp(-net));
}

static double frand(void)
{
    return rand() / (double) RAND_MAX;
}

static void forewards(Tinn t, double* in)
{
    double* x = t.w + t.nhid * t.nips;
    int i;
    for(i = 0; i < t.nhid; i++)
    {
        double sum = 0.0;
        int j;
        for(j = 0; j < t.nips; j++)
        {
            double a = in[j];
            double b = t.w[i * t.nips + j];
            sum += a * b;
        }
        t.h[i] = act(sum + t.b[0]);
    }
    for(i = 0; i < t.nops; i++)
    {
        double sum = 0.0;
        int j;
        for(j = 0; j < t.nhid; j++)
        {
            double a = t.h[j];
            double b = x[i * t.nhid + j];
            sum += a * b;
        }
        t.o[i] = act(sum + t.b[1]);
    }
}

static void twrand(Tinn t)
{
    int wgts = t.nhid * (t.nips + t.nops);
    int i;
    for(i = 0; i < wgts; i++) t.w[i] = frand();
    for(i = 0; i < t.nb; i++) t.b[i] = frand();
}

double xttrain(Tinn t, double* in, double* tg, double rate)
{
    forewards(t, in);
    backwards(t, in, tg, rate);
    return error(t, tg);
}

Tinn xtbuild(int nips, int nhid, int nops)
{
    Tinn t;
    t.nb = 2;
    t.w = (double*) calloc(nhid * (nips + nops), sizeof(*t.w));
    t.b = (double*) calloc(t.nb, sizeof(*t.b));
    t.h = (double*) calloc(nhid, sizeof(*t.h));
    t.o = (double*) calloc(nops, sizeof(*t.o));
    t.nips = nips;
    t.nhid = nhid;
    t.nops = nops;
    srand(time(0));
    twrand(t);
    return t;
}

void xtfree(Tinn t)
{
    free(t.w);
    free(t.h);
    free(t.o);
}
