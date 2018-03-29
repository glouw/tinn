#include "Tinn.h"

#include <stdio.h>
#include <stdlib.h>

int main()
{
    int i;
    int inputs = 2;
    int output = 2;
    int hidden = 2;
    double* I = (double*) calloc(inputs, sizeof(*I));
    double* T = (double*) calloc(output, sizeof(*T));
    Tinn tinn = tnew(inputs, output, hidden);
    /* Input. */
    I[0] = 0.05;
    I[1] = 0.10;
    /* Target. */
    T[0] = 0.01;
    T[1] = 0.99;
    for(i = 0; i < 10000; i++)
    {
        double error = ttrain(tinn, I, T, 0.5);
        printf("error: %0.13f\n", error);
    }
    tfree(tinn);
    free(I);
    free(T);
    return 0;
}
