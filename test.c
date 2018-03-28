#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// [i1]    [h1]    [o1]
//
// [i2]    [h2]    [o2]
//
//     [b1]    [b2]

static double act(const double in)
{
    return 1.0 / (1.0 + exp(-in));
}

int main()
{
    const double rate = 0.5;
    // Input.
    const double i1 = 0.05;
    const double i2 = 0.10;
    // Output.
    const double t1 = 0.01;
    const double t2 = 0.99;
    // Weights and biases.
    double w[] = { 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55 };
    double b[] = { 0.35, 0.60 };

    double et = 0;

    for(int i = 0; i < 10000; i++)
    {
        // Compute.
        const double h1 = act(w[0] * i1 + w[1] * i2 + b[0]);
        const double h2 = act(w[2] * i1 + w[3] * i2 + b[0]);
        const double o1 = act(w[4] * h1 + w[5] * h2 + b[1]);
        const double o2 = act(w[6] * h1 + w[7] * h2 + b[1]);

        // Error calculation.
        const double to1 = t1 - o1;
        const double to2 = t2 - o2;
        const double e1 = 0.5 * to1 * to1;
        const double e2 = 0.5 * to2 * to2;
        et = e1 + e2;

        const double a = -to1 * o1 * (1.0 - o1);
        const double b = -to2 * o2 * (1.0 - o2);
        const double c = (w[4] * a + w[6] * b) * (1.0 - h1);
        const double d = (w[5] * a + w[7] * b) * (1.0 - h2);

        // Back Propogation.
        w[0] -= rate * h1 * c * i1;
        w[1] -= rate * h1 * c * i2;
        w[2] -= rate * h2 * d * i1;
        w[3] -= rate * h2 * d * i2;
        w[4] -= rate * h1 * a;
        w[5] -= rate * h2 * a;
        w[6] -= rate * h1 * b;
        w[7] -= rate * h2 * b;

        #if 0
        printf("w1 %.9f\n", w[0]);
        printf("w2 %.9f\n", w[1]);
        printf("w3 %.9f\n", w[2]);
        printf("w4 %.9f\n", w[3]);
        printf("w5 %.9f\n", w[4]);
        printf("w6 %.9f\n", w[5]);
        printf("w7 %.9f\n", w[6]);
        printf("w8 %.9f\n", w[7]);
        #endif
    }
    printf("%0.12f\n", et);
}
