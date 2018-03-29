#include "Tinn.h"

#include <stdlib.h>
#include <math.h>

static double error(Tinn t, double* T)
{
    double error = 0.0;
    int i;
    for(i = 0; i < t.output; i++)
        error += 0.5 * pow(T[i] - t.O[i], 2.0);
    return error;
}

static void backpass(Tinn t, double* I, double* T, double rate)
{
    int i, j, k;
    double* X = t.W + t.hidden * t.inputs;
    for(i = 0; i < t.inputs; i++)
    {
        double sum = 0.0;
        for(k = 0; k < t.output; k++)
        {
            double a = t.O[k] - T[k];
            double b = t.O[k] * (1 - t.O[k]);
            double c = X[k * t.output + i];
            sum += a * b * c;
        }
        for(j = 0; j < t.hidden; j++)
        {
            double a = sum;
            double b = t.H[i] * (1 - t.H[i]);
            double c = I[j];
            t.W[i * t.hidden + j] -= rate * a * b * c;
        }
    }
    for(i = 0; i < t.output; i++)
    for(j = 0; j < t.hidden; j++)
    {
        double a = t.O[i] - T[i];
        double b = t.O[i] * (1 - t.O[i]);
        double c = t.H[j];
        X[t.hidden * i + j] -= rate * a * b * c;
    }
}

static double act(double net)
{
    return 1.0 / (1.0 + exp(-net));
}

static void forepass(Tinn t, double* I)
{
    int i, j;
    const double B[] = { 0.35, 0.60 };
    double* X = t.W + t.hidden * t.inputs;
    for(i = 0; i < t.hidden; i++)
    {
        double sum = 0.0;
        for(j = 0; j < t.inputs; j++)
        {
            double a = I[j];
            double b = t.W[i * t.inputs + j];
            sum += a * b;
        }
        t.H[i] = act(sum + B[0]);
    }
    for(i = 0; i < t.output; i++)
    {
        double sum = 0.0;
        for(j = 0; j < t.hidden; j++)
        {
            double a = t.H[j];
            double b = X[i * t.hidden + j];
            sum += a * b;
        }
        t.O[i] = act(sum + B[1]);
    }
}

double ttrain(Tinn t, double* I, double* T, double rate)
{
    forepass(t, I);
    backpass(t, I, T, rate);
    return error(t, T);
}

Tinn tnew(int inputs, int output, int hidden)
{
    Tinn t;
    t.inputs = inputs;
    t.output = output;
    t.hidden = hidden;
    t.H = (double*) calloc(hidden, sizeof(*t.H));
    t.O = (double*) calloc(output, sizeof(*t.O));
    t.W = (double*) calloc(hidden * (inputs + output), sizeof(*t.W));
    t.W[0] = 0.15;
    t.W[1] = 0.20;
    t.W[2] = 0.25;
    t.W[3] = 0.30;
    t.W[4] = 0.40;
    t.W[5] = 0.45;
    t.W[6] = 0.50;
    t.W[7] = 0.55;
    return t;
}

void tfree(Tinn t)
{
    free(t.W);
    free(t.H);
    free(t.O);
}
