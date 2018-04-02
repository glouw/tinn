![](img/logo.PNG)

Tinn (Tiny Neural Network) is a 200 line dependency free neural network library written in C99.
Tinn can be compiled with any C++ compiler as well.

    #include "Tinn.h"
    #include <stdio.h>

    #define len(a) ((int) (sizeof(a) / sizeof(*a)))

    int main()
    {
        float in[] = { 0.05, 0.10 };
        float tg[] = { 0.01, 0.99 };
        /* Two hidden neurons */
        const Tinn tinn = xtbuild(len(in), 2, len(tg));
        for(int i = 0; i < 1000; i++)
        {
            float error = xttrain(tinn, in, tg, 0.5);
            printf("%.12f\n", error);
        }
        xtfree(tinn);
        return 0;
    }

For a quick demo, get some training data:

    wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data

And if you're on Linux / MacOS just build and run:

    make; ./tinn

If you're on Windows it's:

    mingw32-make & tinn.exe
