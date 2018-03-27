// This program uses the Genann Neural Network Library to learn hand written digits.
//
// Get it from the machine learning database:
//     wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data

#include "Genann.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define toss(t, n) ((t*) malloc((n) * sizeof(t)))

#define retoss(ptr, t, n) (ptr = (t*) realloc((ptr), (n) * sizeof(t)))

typedef struct
{
    double** id;
    double** od;
    int icols;
    int ocols;
    int rows;
    int split;
}
Data;

static int lns(FILE* const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while((ch = getc(file)) != EOF)
    {
        if(ch == '\n')
            lines++;
        pc = ch;
    }
    if(pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

static char* readln(FILE* const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char* line = toss(char, size);
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
            retoss(line, char, size *= 2);
    }
    line[reads] = '\0';
    return line;
}

static double** new2d(const int rows, const int cols)
{
    double** row = toss(double*, rows);
    for(int r = 0; r < rows; r++)
        row[r] = toss(double, cols);
    return row;
}

static Data ndata(const int icols, const int ocols, const int rows, const double percentage)
{
    const Data data = {
        new2d(rows, icols), new2d(rows, ocols), icols, ocols, rows, (int) (rows * percentage)
    };
    return data;
}

static void parse(const Data data, char* line, const int row)
{
    const int cols = data.icols + data.ocols;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < data.icols)
            data.id[row][col] = val;
        else
            data.od[row][col - data.icols] = val;
    }
}

static void dfree(const Data d)
{
    for(int row = 0; row < d.rows; row++)
    {
        free(d.id[row]);
        free(d.od[row]);
    }
    free(d.id);
    free(d.od);
}

static void shuffle(const Data d, const int upper)
{
    for(int a = 0; a < upper; a++)
    {
        const int b = rand() % d.split;
        double* ot = d.od[a];
        double* it = d.id[a];
        // Swap output.
        d.od[a] = d.od[b];
        d.od[b] = ot;
        // Swap input.
        d.id[a] = d.id[b];
        d.id[b] = it;
    }
}

static void print(const double* const arr, const int size)
{
    for(int i = 0; i < size; i++)
        printf("%d ", arr[i] > 0.9);
}

static int cmp(const double* const a, const double* const b, const int size)
{
    for(int i = 0; i < size; i++)
    {
        const int aa = a[i] > 0.9;
        const int bb = b[i] > 0.9;
        if(aa != bb)
            return 0;
    }
    return 1;
}

static void predict(Genann* ann, const Data d)
{
    int matches = 0;
    for(int i = d.split; i < d.rows; i++)
    {
        // Prediciton.
        const double* const pred = genann_run(ann, d.id[i]);
        const double* const real = d.od[i];
        print(pred, d.ocols);
        printf(":: ");
        print(real, d.ocols);
        const int match = cmp(pred, real, d.ocols);
        printf("-> %d\n", match);
        matches += match;
    }
    printf("%f\n", (double) matches / (d.rows - d.split));
}

static Data build(const char* path, const int icols, const int ocols, const double percentage)
{
    FILE* file = fopen(path, "r");
    const int rows = lns(file);
    Data data = ndata(icols, ocols, rows, percentage);
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

static Genann* train(const Data d, const int ntimes, const int layers, const int neurons, const double rate)
{
    Genann* const ann = genann_init(d.icols, layers, neurons, d.ocols);
    double annealed = rate;
    for(int i = 0; i < ntimes; i++)
    {
        shuffle(d, d.split);
        for(int j = 0; j < d.split; j++)
            genann_train(ann, d.id[j], d.od[j], annealed);
        printf("%f: %f\n", (double) i / ntimes, annealed);
        annealed *= 0.95;
    }
    return ann;
}

int main(int argc, char* argv[])
{
    (void) argc;
    (void) argv;
    srand(time(0));
    const Data data = build("semeion.data", 256, 10, 0.9);
    shuffle(data, data.rows);
    Genann* ann = train(data, 128, 1, data.icols / 2.0, 3.0); // Hyperparams.
    predict(ann, data);
    genann_free(ann);
    dfree(data);
    return 0;
}
