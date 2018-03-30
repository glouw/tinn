#pragma once

typedef struct
{
    double* w; // Weights.
    double* b; // Biases.
    double* h; // Hidden layer.
    double* o; // Output layer.

    // Number of biases - always two - Tinn only supports a single hidden layer.
    int nb;

    // Number of weights.
    int nw;

    int nips; // Number of inputs.
    int nhid; // Number of hidden neurons.
    int nops; // Number of outputs.
}
Tinn;

// Trains a tinn with an input and target output with a learning rate.
// Returns error rate of the neural network.
double xttrain(const Tinn, const double* in, const double* tg, double rate);

// Builds a new tinn object given number of inputs (nips),
// number of hidden neurons for the hidden layer (nhid),
// and number of outputs (nops).
Tinn xtbuild(int nips, int nhid, int nops);

// Returns an output prediction given an input.
double* xpredict(const Tinn, const double* in);

// Saves the tinn to disk.
void xtsave(const Tinn, const char* path);

// Loads a new tinn from disk.
Tinn xtload(const char* path);

// Frees a tinn from the heap.
void xtfree(const Tinn);
