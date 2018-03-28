#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double act(const double in)
{
    return 1.0 / (1.0 + exp(-in));
}

static double shid(const double W[], const double I[], const int neuron, const int inputs)
{
    double sum = 0.0;
    int i;
    for(i = 0; i < inputs; i++)
        sum += I[i] * W[i + neuron * inputs];
    return sum;
}

static double sout(const double W[], const double I[], const int neuron, const int inputs, const int hidden)
{
    double sum = 0.0;
    int i;
    for(i = 0; i < inputs; i++)
        sum += I[i] * W[i + hidden * (neuron + inputs)];
    return sum;
}

static double cerr(const double T[], const double O[], const int count)
{
    double ssqr = 0.0;
    int i;
    for(i = 0; i < count; i++)
    {
        const double sub = T[i] - O[i];
        ssqr += sub * sub;
    }
    return 0.5 * ssqr;
}

static void bprop(double W[], const double I[], const double H[], const double O[], const double T[], const double rate)
{
    const double a = -(T[0] - O[0]) * O[0] * (1.0 - O[0]);
    const double b = -(T[1] - O[1]) * O[1] * (1.0 - O[1]);
    const double c = (W[4] * a + W[6] * b) * (1.0 - H[0]);
    const double d = (W[5] * a + W[7] * b) * (1.0 - H[1]);
    /* Hidden layer */
    W[0] -= rate * H[0] * c * I[0];
    W[1] -= rate * H[0] * c * I[1];
    W[2] -= rate * H[1] * d * I[0];
    W[3] -= rate * H[1] * d * I[1];
    /* Output layer */
    W[4] -= rate * H[0] * a;
    W[5] -= rate * H[1] * a;
    W[6] -= rate * H[0] * b;
    W[7] -= rate * H[1] * b;
}

/* Single layer feed forward neural network with back propogation error correction */
static double train(const double I[], const double T[], const int nips, const int nops, const double rate, const int iters)
{
    const double B[] = { 0.35, 0.60 };
    const int nhid = sizeof(B) / sizeof(*B);
    double W[] = { 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55 };
    double* H = (double*) malloc(sizeof(*H) * nhid);
    double* O = (double*) malloc(sizeof(*O) * nops);
    double error;
    int iter;
    for(iter = 0; iter < iters; iter++)
    {
        int i;
        for(i = 0; i < nhid; i++) H[i] = act(B[0] + shid(W, I, i, nips));
        for(i = 0; i < nops; i++) O[i] = act(B[1] + sout(W, H, i, nips, nhid));
        bprop(W, I, H, O, T, rate);
    }
    error = cerr(T, O, nops);
    free(H);
    free(O);
    return error;
}

int main()
{
    const double rate = 0.5;

    const double I[] = { 0.05, 0.10 };

    const double T[] = { 0.01, 0.99 };

    const double error = train(I, T, sizeof(I) / sizeof(*I), sizeof(T) / sizeof(*T), rate, 10000);

    printf("%f\n", error);

    return 0;
}
