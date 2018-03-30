#pragma once

typedef struct
{
    double* w; // Weights.
    double* b; // Biases.
    double* h; // Hidden layer.
    double* o; // Output layer.

    // Number of biases - always two - Tinn only supports a single hidden layer.
    int nb;

    int nips; // Number of inputs.
    int nhid; // Number of hidden neurons.
    int nops; // Number of outputs.
}
Tinn;

double xttrain(const Tinn, const double* in, const double* tg, double rate);

Tinn xtbuild(int nips, int nhid, int nops);

void xtfree(Tinn);

double* xpredict(const Tinn, const double* in);
