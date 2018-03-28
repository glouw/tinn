#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static double act(double net)
{
    return 1.0 / (1.0 + exp(-net));
}

static void forepass(double* I, double* O, double* H, double* W, double* B, const int inputs, const int output, const int hidden)
{
    double* X = W + hidden * inputs;
    for(int i = 0; i < hidden; i++) { for(int j = 0; j < inputs; j++) H[i] += I[j] * W[i * inputs + j]; H[i] = act(H[i] + B[0]); }
    for(int i = 0; i < output; i++) { for(int j = 0; j < hidden; j++) O[i] += H[j] * X[i * hidden + j]; O[i] = act(O[i] + B[1]); }
}

static void backpass(double* I, double* O, double* H, double* W, double* T, const int inputs, const int output, const int hidden, const double rate)
{
    double* X = W + hidden * inputs;
    for(int i = 0; i < output; i++)
    for(int j = 0; j < hidden; j++)
        X[2 * i + j] -= rate * ((O[i] - T[i]) * (O[i] * (1 - O[i])) * H[j]);

    //W[4] -= rate * ((T[0] - O[0]) * (T[0] * (1 - T[0])) * H[0]);
    //W[5] -= rate * ((T[0] - O[0]) * (T[0] * (1 - T[0])) * H[1]);
    //W[6] -= rate * ((T[1] - O[1]) * (T[1] * (1 - T[1])) * H[0]);
    //W[7] -= rate * ((T[1] - O[1]) * (T[1] * (1 - T[1])) * H[1]);
}

static double cerror(double *O, double* T, const int output)
{
    double error = 0.0;
    for(int i = 0; i < output; i++)
        error += 0.5 * pow(T[i] - O[i], 2.0);
    return error;
}

static double* train(double* I, double* T, const int inputs, const int output, const int hidden)
{
    // Weights.
    double* W = (double*) calloc(hidden * (inputs + output), sizeof(*W));
    W[0] = 0.15;
    W[1] = 0.20;
    W[2] = 0.25;
    W[3] = 0.30;
    W[4] = 0.40;
    W[5] = 0.45;
    W[6] = 0.50;
    W[7] = 0.55;

    // Fixed at single hidden layer - only two biases are needed.
    double B[] = { 0.35, 0.60 };

    // Hidden layer.
    double* H = (double*) calloc(hidden, sizeof(*H));

    // Output layer. Will eventually converge to output with enough iterations.
    double* O = (double*) calloc(output, sizeof(*O));

    // Computes hidden and target nodes.
    forepass(I, O, H, W, B, inputs, output, hidden);

    // Computes output to target error.
    double err = cerror(O, O, output);

    printf("error: %f\n", err);

    // Updates weights based on target error.
    backpass(I, O, H, W, T, inputs, output, hidden, 0.5);

    printf("W5: %f\n", W[4]);
    printf("W6: %f\n", W[5]);
    printf("W7: %f\n", W[6]);
    printf("W8: %f\n", W[7]);

    printf("%f\n", H[0]);
    printf("%f\n", H[1]);
    printf("%f\n", O[0]);
    printf("%f\n", O[1]);

    free(H);

    return W;
}

double* predict(double* I, double* W, const int inputs, const int output)
{
    double* O = NULL;

    // ...

    return O;
}

int main()
{
    const int inputs = 2, output = 2, hidden = 2;

    // Input.
    double* I = (double*) calloc(inputs, sizeof(*I));
    I[0] = 0.05;
    I[1] = 0.10;

    // Target.
    double* T = (double*) calloc(output, sizeof(*I));
    T[0] = 0.01;
    T[1] = 0.99;

    train(I, T, inputs, output, hidden);

    return 0;
}
