#include "Tinn.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Error function.
static double err(double a, double b)
{
    return 0.5 * pow(a - b, 2.0);
}

// Partial derivative of error function.
static double pderr(double a, double b)
{
    return a - b;
}

// Total error.
static double terr(const double* tg, const double* o, int size)
{
    double sum = 0.0;
    for(int i = 0; i < size; i++)
        sum += err(tg[i], o[i]);
    return sum;
}

// Activation function.
static double act(double a)
{
    return 1.0 / (1.0 + exp(-a));
}

// Partial derivative of activation function.
static double pdact(double a)
{
    return a * (1.0 - a);
}

// Floating point random from 0.0 - 1.0.
static double frand()
{
    return rand() / (double) RAND_MAX;
}

// Back propagation.
static void backwards(const Tinn t, const double* in, const double* tg, double rate)
{
    for(int i = 0; i < t.nhid; i++)
    {
        double sum = 0.0;
        // Calculate total error change with respect to output.
        for(int j = 0; j < t.nops; j++)
        {
            double a = pderr(t.o[j], tg[j]);
            double b = pdact(t.o[j]);
            sum += a * b * t.x[j * t.nhid + i];
            // Correct weights in hidden to output layer.
            t.x[j * t.nhid + i] -= rate * a * b * t.h[i];
        }
        // Correct weights in input to hidden layer.
        for(int j = 0; j < t.nips; j++)
            t.w[i * t.nips + j] -= rate * sum * pdact(t.h[i]) * in[j];
    }
}

// Forward propagation.
static void forwards(const Tinn t, const double* in)
{
    // Calculate hidden layer neuron values.
    for(int i = 0; i < t.nhid; i++)
    {
        double sum = 0.0;
        for(int j = 0; j < t.nips; j++)
            sum += in[j] * t.w[i * t.nips + j];
        t.h[i] = act(sum + t.b[0]);
    }
    // Calculate output layer neuron values.
    for(int i = 0; i < t.nops; i++)
    {
        double sum = 0.0;
        for(int j = 0; j < t.nhid; j++)
            sum += t.h[j] * t.x[i * t.nhid + j];
        t.o[i] = act(sum + t.b[1]);
    }
}

// Randomizes weights and biases.
static void twrand(const Tinn t)
{
    for(int i = 0; i < t.nw; i++) t.w[i] = frand() - 0.5;
    for(int i = 0; i < t.nb; i++) t.b[i] = frand() - 0.5;
}

double* xpredict(const Tinn t, const double* in)
{
    forewards(t, in);
    return t.o;
}

double xttrain(const Tinn t, const double* in, const double* tg, double rate)
{
    forewards(t, in);
    backwards(t, in, tg, rate);
    return terr(tg, t.o, t.nops);
}

Tinn xtbuild(int nips, int nhid, int nops)
{
    Tinn t;
    // Tinn only supports one hidden layer so there are two biases.
    t.nb = 2;
    t.nw = nhid * (nips + nops);
    t.w = (double*) calloc(t.nw, sizeof(*t.w));
    t.x = t.w + nhid * nips;
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

void xtsave(const Tinn t, const char* path)
{
    FILE* file = fopen(path, "w");
    // Header.
    fprintf(file, "%d %d %d\n", t.nips, t.nhid, t.nops);
    // Biases and weights.
    for(int i = 0; i < t.nb; i++) fprintf(file, "%lf\n", t.b[i]);
    for(int i = 0; i < t.nw; i++) fprintf(file, "%lf\n", t.w[i]);
    fclose(file);
}

Tinn xtload(const char* path)
{
    FILE* file = fopen(path, "r");
    int nips = 0;
    int nhid = 0;
    int nops = 0;
    // Header.
    fscanf(file, "%d %d %d\n", &nips, &nhid, &nops);
    // A new tinn is returned.
    Tinn t = xtbuild(nips, nhid, nips);
    // Biases and weights.
    for(int i = 0; i < t.nb; i++) fscanf(file, "%lf\n", &t.b[i]);
    for(int i = 0; i < t.nw; i++) fscanf(file, "%lf\n", &t.w[i]);
    fclose(file);
    return t;
}

void xtfree(const Tinn t)
{
    free(t.w);
    free(t.b);
    free(t.h);
    free(t.o);
}
