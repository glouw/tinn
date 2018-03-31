#include "Tinn.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Error function.
static float err(float a, float b)
{
    return 0.5f * powf(a - b, 2.0f);
}

// Partial derivative of error function.
static float pderr(float a, float b)
{
    return a - b;
}

// Total error.
static float terr(const float* tg, const float* o, int size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; i++)
        sum += err(tg[i], o[i]);
    return sum;
}

// Activation function.
static float act(float a)
{
    return 1.0f / (1.0f + expf(-a));
}

// Partial derivative of activation function.
static float pdact(float a)
{
    return a * (1.0f - a);
}

// Floating point random from 0.0 - 1.0.
static float frand()
{
    return rand() / (float) RAND_MAX;
}

// Back propagation.
static void backwards(const Tinn t, const float* in, const float* tg, float rate)
{
    for(int i = 0; i < t.nhid; i++)
    {
        float sum = 0.0f;
        // Calculate total error change with respect to output.
        for(int j = 0; j < t.nops; j++)
        {
            float a = pderr(t.o[j], tg[j]);
            float b = pdact(t.o[j]);
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
static void forewards(const Tinn t, const float* in)
{
    // Calculate hidden layer neuron values.
    for(int i = 0; i < t.nhid; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < t.nips; j++)
            sum += in[j] * t.w[i * t.nips + j];
        t.h[i] = act(sum + t.b[0]);
    }
    // Calculate output layer neuron values.
    for(int i = 0; i < t.nops; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < t.nhid; j++)
            sum += t.h[j] * t.x[i * t.nhid + j];
        t.o[i] = act(sum + t.b[1]);
    }
}

// Randomizes weights and biases.
static void twrand(const Tinn t)
{
    for(int i = 0; i < t.nw; i++) t.w[i] = frand() - 0.5f;
    for(int i = 0; i < t.nb; i++) t.b[i] = frand() - 0.5f;
}

float* xpredict(const Tinn t, const float* in)
{
    forewards(t, in);
    return t.o;
}

float xttrain(const Tinn t, const float* in, const float* tg, float rate)
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
    t.w = (float*) calloc(t.nw, sizeof(*t.w));
    t.x = t.w + nhid * nips;
    t.b = (float*) calloc(t.nb, sizeof(*t.b));
    t.h = (float*) calloc(nhid, sizeof(*t.h));
    t.o = (float*) calloc(nops, sizeof(*t.o));
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
    for(int i = 0; i < t.nb; i++) fprintf(file, "%f\n", (double) t.b[i]);
    for(int i = 0; i < t.nw; i++) fprintf(file, "%f\n", (double) t.w[i]);
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
    for(int i = 0; i < t.nb; i++) fscanf(file, "%f\n", &t.b[i]);
    for(int i = 0; i < t.nw; i++) fscanf(file, "%f\n", &t.w[i]);
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
