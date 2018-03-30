![](img/logo.PNG)

Tinn (Tiny Neural Network) is a dependency free ANSI-C neural network library.

    #include "Tinn.h"
    #include <stdio.h>

    #define len(a) ((int) (sizeof(a) / sizeof(*a)))

    int main(void)
    {
        double in[] = { 0.05, 0.10 };
        double tg[] = { 0.01, 0.99 };
        /* Two hidden nuerons */
        const Tinn tinn = xtbuild(len(in), 2, len(tg));
        int i;
        for(i = 0; i < 10000; i++)
        {
            double error = xttrain(tinn, in, tg, 0.5);
            printf("%.12f\n", error);
        }
        xtfree(tinn);
        return 0;
    }

For a demo, get some data:

    wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data

And then build and run the test sample:

    make; ./tinn
