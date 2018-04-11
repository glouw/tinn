![](img/logo.PNG)

Tinn (Tiny Neural Network) is a 200 line dependency free neural network library written in C99.

For a demo on how to learn hand written digits, get some training data:

    wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data

And if you're on Linux / MacOS just build and run Tinn with the test file:

    make; ./tinn

If you're on Windows it's:

    mingw32-make & tinn.exe

The training data consists of hand written digits written both slowly and quickly.
Each line in the data set corresponds to one handwritten digit. Each digit is 16x16 pixels in size
giving 256 inputs to the neural network.

At the end of the line 10 digits signify the hand written digit:

    0: 1 0 0 0 0 0 0 0 0 0
    1: 0 1 0 0 0 0 0 0 0 0
    2: 0 0 1 0 0 0 0 0 0 0
    3: 0 0 0 1 0 0 0 0 0 0
    4: 0 0 0 0 1 0 0 0 0 0
    ...
    9: 0 0 0 0 0 0 0 0 0 1

This gives 10 outputs to the neural network. The test program will output the
accuracy for each digit. Expect above 99% accuracy for the correct digit, and
less that 0.1% accuracy for the other digits.

# Features

* Portable - Runs on Windows, MacOS, Linux, and embedded chips like ARM, AVR, and Microchip

* Sigmoidal activation.

* One hidden layer.

# Tips

* Tinn will never use more than the C standard library.

* Tinn is great for embedded systems. Train a model on your powerful desktop and load
it onto a microcontroller and use the analog to digital converter to predict real time events.

* The Tinn source code will always be less than 200 lines. Functions externed in the Tinn header
are protected with the _xt_ namespace standing for _externed tinn_.

* Tinn can easily be multi-threaded with a bit of ingenuity but the master branch will remain
single threaded to aid development for embedded systems.

* Tinn does not seed the random number generator. Do not forget to do so yourself.

* Always shuffle your input data. Shuffle again after every training iteration.

* Get greater training accuracy by annealing your learning rate. For instance, multiply
your learning rate by 0.99 every training iteration. This will zero in on a good learning minima.

# Disclaimer

Tinn is a practice in minimalism.

Tinn is not a fully featured neural network C library like Kann, or Genann:

    https://github.com/attractivechaos/kann

    https://github.com/codeplea/genann
