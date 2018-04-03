#pragma once

typedef struct
{
    float* w; // All the weights.
    float* x; // Hidden to output layer weights.
    float* b; // Biases.
    float* h; // Hidden layer.
    float* o; // Output layer.

    int nb; // Number of biases - always two - Tinn only supports a single hidden layer.
    int nw; // Number of weights.

    int nips; // Number of inputs.
    int nhid; // Number of hidden neurons.
    int nops; // Number of outputs.
}
Tinn;

// Trains a tinn with an input and target output with a learning rate.
// Returns error rate of the neural network.
float xttrain(Tinn, const float* in, const float* tg, float rate);

// Builds a new tinn object given number of inputs (nips),
// number of hidden neurons for the hidden layer (nhid),
// and number of outputs (nops).
Tinn xtbuild(int nips, int nhid, int nops);

// Returns an output prediction given an input.
float* xtpredict(Tinn, const float* in);

// Saves the tinn to disk.
void xtsave(Tinn, const char* path);

// Loads a new tinn from disk.
Tinn xtload(const char* path);

// Frees a tinn from the heap.
void xtfree(Tinn);
