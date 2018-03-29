#include "Tinn.h"

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

static Data ndata(const int icols, const int ocols, const int rows)
{
    const Data data = {
        new2d(rows, icols), new2d(rows, ocols), icols, ocols, rows
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

static void shuffle(const Data d)
{
    for(int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
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

static Data build(const char* path, const int icols, const int ocols)
{
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        printf("Could not open %s\n", path);
        printf("Get the training data: \n");
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(icols, ocols, rows);
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

int main(void)
{
    const Data data = build("semeion.data", 256, 10);
    shuffle(data);
    const Tinn tinn = xtbuild(data.icols, 64, data.ocols);
    for(int i = 0; i < 10000; i++)
    {
        double error = 0.0;
        for(int j = 0; j < data.rows; j++)
        {
            double* in = data.id[j];
            double* tg = data.od[j];
            //error += xttrain(tinn, in, tg, 0.5);
        }
        printf("%.12f\n", error);
    }
    xtfree(tinn);
    dfree(data);
    return 0;
}
