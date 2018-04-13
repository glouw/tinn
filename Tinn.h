#pragma once

typedef struct
{
    // All the weights.
    float* w;
    // Hidden to output layer weights.
    float* x;
    // Biases.
    float* b;
    // Hidden layer.
    float* h;
    // Output layer.
    float* o;
    // Number of biases - always two - Tinn only supports a single hidden layer.
    int nb;
    // Number of weights.
    int nw;
    // Number of inputs.
    int nips;
    // Number of hidden neurons.
    int nhid;
    // Number of outputs.
    int nops;
}
Tinn;

float* xtpredict(Tinn, const float* in);

float xttrain(Tinn, const float* in, const float* tg, float rate);

Tinn xtbuild(int nips, int nhid, int nops);

void xtsave(Tinn, const char* path);

Tinn xtload(const char* path);

void xtfree(Tinn);

void xtprint(const float* arr, const int size);
