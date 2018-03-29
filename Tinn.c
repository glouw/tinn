#include "Tinn.h"

#include <stdlib.h>
#include <math.h>

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
    int i, j, k;
    double* x = t.w + t.nhid * t.nips;
    for(i = 0; i < t.nhid; i++)
    {
        double sum = 0.0;
        for(k = 0; k < t.nops; k++)
        {
            double a = t.o[k] - tg[k];
            double b = t.o[k] * (1 - t.o[k]);
            double c = x[k * t.nhid + i];
            sum += a * b * c;
        }
        for(j = 0; j < t.nips; j++)
        {
            double a = sum;
            double b = t.h[i] * (1 - t.h[i]);
            double c = in[j];
            t.w[i * t.nips + j] -= rate * a * b * c;
        }
    }
    for(i = 0; i < t.nops; i++)
    for(j = 0; j < t.nhid; j++)
    {
        double a = t.o[i] - tg[i];
        double b = t.o[i] * (1 - t.o[i]);
        double c = t.h[j];
        x[t.nhid * i + j] -= rate * a * b * c;
    }
}

static double act(double net)
{
    return 1.0 / (1.0 + exp(-net));
}

static void forewards(Tinn t, double* in)
{
    int i, j;
    const double bias[] = { 0.35, 0.60 };
    double* x = t.w + t.nhid * t.nips;
    for(i = 0; i < t.nhid; i++)
    {
        double sum = 0.0;
        for(j = 0; j < t.nips; j++)
        {
            double a = in[j];
            double b = t.w[i * t.nips + j];
            sum += a * b;
        }
        t.h[i] = act(sum + bias[0]);
    }
    for(i = 0; i < t.nops; i++)
    {
        double sum = 0.0;
        for(j = 0; j < t.nhid; j++)
        {
            double a = t.h[j];
            double b = x[i * t.nhid + j];
            sum += a * b;
        }
        t.o[i] = act(sum + bias[1]);
    }
}

static void twrand(Tinn t)
{
#if 0
    t.w[0] = 0.15;
    t.w[1] = 0.20;
    t.w[2] = 0.25;
    t.w[3] = 0.30;

    t.w[4] = 0.40;
    t.w[5] = 0.45;
    t.w[6] = 0.50;
    t.w[7] = 0.55;
#else
    t.w[0] = 0.15;
    t.w[1] = 0.20;
    t.w[2] = 0.25;
    t.w[3] = 0.30;
    t.w[4] = 0.30;
    t.w[5] = 0.30;

    t.w[6] = 0.40;
    t.w[7] = 0.45;
    t.w[8] = 0.50;
    t.w[9] = 0.55;
    t.w[10] = 0.55;
    t.w[11] = 0.55;
#endif
}

double ttrain(Tinn t, double* in, double* tg, double rate)
{
    forewards(t, in);
    backwards(t, in, tg, rate);
    return error(t, tg);
}

Tinn tbuild(int nips, int nops, int nhid)
{
    Tinn t;
    t.o = (double*) calloc(nops, sizeof(*t.o));
    t.h = (double*) calloc(nhid, sizeof(*t.h));
    t.w = (double*) calloc(nhid * (nips + nops), sizeof(*t.w));
    t.nops = nops;
    t.nhid = nhid;
    t.nips = nips;
    twrand(t);
    return t;
}

void tfree(Tinn t)
{
    free(t.w);
    free(t.h);
    free(t.o);
}
