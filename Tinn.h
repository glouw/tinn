#ifndef _TINN_H_
#define _TINN_H_

typedef struct
{
    double* o; /*       Output layer       */
    double* h; /*       Hidden layer       */
    double* w; /*     Training weights     */
    int nops;  /* Number of Output Neurons */
    int nhid;  /* Number of Hidden Neurons */
    int nips;  /* Number of Input  Neurons */
}
Tinn;

/* Trains a Tinn object given input (in) data, target (tg) data,
 * and a learning rate (recommended 0.0 - 1.0) */
double ttrain(Tinn, double* in, double* tg, double rate);

/* Returns a Tinn object given number of inputs (nips),
 * number of outputs (nops), and number of hidden layers (nhid) */
Tinn tbuild(int nips, int nops, int nhid);

/* Frees a tinn object from heap memory */
void tfree(Tinn);

#endif
