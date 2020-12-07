#include "Tinn.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

typedef struct
{
    float** in;
    float** tg;
    int nips;
    int nops;
    int rows;
}
Data;

typedef struct {
    int k;
    float tg;
    float pd;
} pos;

void output_svg(int j, Data data, int realnum, float pcage, int goodbad)
{
	printf("ln -s %05d-%d.svg %s/\n", j, realnum, goodbad ? "good" : "bad");
}

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
    char* line = (char*) malloc((size) * sizeof(char));
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
            line = (char*) realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

static float** new2d(const int rows, const int cols)
{
    float** row = (float**) malloc((rows) * sizeof(float*));
    for(int r = 0; r < rows; r++)
        row[r] = (float*) malloc((cols) * sizeof(float));
    return row;
}

static Data ndata(const int nips, const int nops, const int rows)
{
    const Data data = {
        new2d(rows, nips), new2d(rows, nops), nips, nops, rows
    };
    return data;
}

static void parse(const Data data, char* line, const int row)
{
    for(int col = 0; col < data.nips; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, ", "));
        data.in[row][col] = val/100.0;
    }
    const float val = atof(strtok(NULL, ", "));
    for(int col = 0; col < data.nops; col++) {
        data.tg[row][col] = 0.0;
    }
    data.tg[row][(int)val] = 1.0;
}

static void dfree(const Data d)
{
    for(int row = 0; row < d.rows; row++)
    {
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}

static void shuffle(const Data d)
{
    for(int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
        float* ot = d.tg[a];
        float* it = d.in[a];
        // Swap output.
        d.tg[a] = d.tg[b];
        d.tg[b] = ot;
        // Swap input.
        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}

static Data build(const char* path, const int nips, const int nops)
{
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        printf("Could not open %s\n", path);
        printf("Get it from the machine learning database: ");
        printf("wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data\n");
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(nips, nops, rows);
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

int sort_by_pd(const void *a, const void *b) {
    pos x = *(pos*)a;
    pos y = *(pos*)b;
    if (x.pd > y.pd) { return -1; }
    if (x.pd < y.pd) { return +1; }
    return 0;
}

int main()
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    // Input and output size is harded coded here as machine learning
    // repositories usually don't include the input and output size in the data itself.
    const int nips = 16;
    const int nops = 10;
    // Load the training set.
    const Data data = build("pendigits.tes", nips, nops);
    // This is how you load the neural network from disk.
    const Tinn loaded = xtload("saved.tinn");
    pos check[nops];
    int correct = 0;

    // Now we do a prediction with the neural network we loaded from disk.
    for (int j = 0; j < data.rows; j++) {
        const float* const in = data.in[j];
        const float* const tg = data.tg[j];
        const float* const pd = xtpredict(loaded, in);
        // To find the "best match", we need to sort by probability (`pd`)
        // whilst keeping the target (`tg`) aligned.  Copying them into
        // our struct and then `qsort`ing on `pd` satisfies this.
        for(int i = 0; i < data.nops; i++) {
            check[i].k = i;
            check[i].tg = tg[i];
            check[i].pd = pd[i];
        }
        qsort(check, data.nops, sizeof(pos), sort_by_pd);
        // If the highest probability guess is the correct one, success.
        if (check[0].tg == 1) {
            correct++;
        }
        // Otherwise we print out our best guess and the correct answer.
        else {
            int realnum = -1;
            printf("%05d %d %.5f | ", j, check[0].k, (double) check[0].pd);
            for (int i=1; i < data.nops; i++) {
                if (check[i].tg == 1) {
                    printf("%d %.5f", check[i].k, (double) check[i].pd);
                    realnum = i;
                }
            }
            printf("\n");
        }
    }
    // 
    printf("%d correct out of %d rows\n", correct, data.rows);
    // All done. Let's clean up.
    xtfree(loaded);
    dfree(data);
    return 0;
}
