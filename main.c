// This program uses the Genann Neural Network Library to learn hand written digits.
//
// Get it from the machine learning database:
//     wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genann.h"

#define toss(t, n) ((t*) malloc((n) * sizeof(t)))

#define retoss(ptr, t, n) (ptr = (t*) realloc((ptr), (n) * sizeof(t)))

typedef struct
{
    double** id;
    double** od;
    int icols;
    int ocols;
    int rows;
}
Data;

static int flns(FILE* const file)
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

static char* freadln(FILE* const file)
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

static Data ndata(const int icols, const int ocols, const int rows)
{
    const Data data = { new2d(rows, icols), new2d(rows, ocols), icols, ocols, rows };
    return data;
}

static void dparse(const Data data, char* line, const int row)
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

static genann* dtrain(const Data d, const int ntimes, const int layers, const int neurons, const int rate)
{
    genann* const ann = genann_init(d.icols, layers, neurons, d.ocols);
    for(int i = 0; i < ntimes; i++)
    for(int j = 0; j < d.rows; j++)
        genann_train(ann, d.id[j], d.od[j], rate);
    return ann;
}

static void dpredict(genann* ann, const Data d)
{
    for(int i = 0; i < d.rows; i++)
    {
        const double* const pred = genann_run(ann, d.id[i]);
        for(int j = 0; j < d.ocols; j++)
            printf("%s%d", j > 0 ? " " : "", pred[j] > 0.9);
        putchar('\n');
    }
}

static Data dbuild(char* path, const int icols, const int ocols)
{
    FILE* file = fopen(path, "r");
    const int rows = flns(file);
    Data data = ndata(icols, ocols, rows);
    for(int row = 0; row < rows; row++)
    {
        char* line = freadln(file);
        dparse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

int main(int argc, char* argv[])
{
    (void) argc;
    (void) argv;
    const Data data = dbuild("semeion.data", 256, 10);
    genann* ann = dtrain(data, 256, 1, 32, 1);
    dpredict(ann, data);
    genann_free(ann);
    dfree(data);
    return 0;
}
