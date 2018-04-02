#include "Tinn.h"
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <SDL2/SDL.h>

typedef struct
{
    bool down;
    int x;
    int y;
}
Input;

typedef struct
{
    float** in;
    float** tg;
    int nips;
    int nops;
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
    char* line = ((char*) malloc((size) * sizeof(char)));
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
    const int cols = data.nips + data.nops;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < data.nips)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.nips] = val;
    }
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

void dprint(const float* const p, const int size)
{
    for(int i = 0; i < size; i++)
        printf("%f ", (double) p[i]);
    printf("\n");
}

typedef struct
{
    int i;
    float val;
}
Index;

Index ixmax(const float* const p, const int size)
{
    Index ix;
    ix.val = -FLT_MAX;
    for(int i = 0; i < size; i++)
        if(p[i] > ix.val)
            ix.val = p[ix.i = i];
    return ix;
}

void dploop(const Tinn tinn, const Data data)
{
    SDL_Renderer* renderer;
    SDL_Window* window;
    #define W 16
    #define H 16
    #define S 20
    const int xres = W * S;
    const int yres = H * S;
    SDL_CreateWindowAndRenderer(xres, yres, 0, &window, &renderer);
    static float digit[W * H];
    Input input = { false, 0, 0 };
    for(SDL_Event e; true; SDL_PollEvent(&e))
    {
        if(e.type == SDL_QUIT)
            exit(1);
        const int button = SDL_GetMouseState(&input.x, &input.y);
        // Draw digit.
        if(button)
        {
            const int xx = input.x / S;
            const int yy = input.y / S;
            const int w = 2;
            for(int i = 0; i < w; i++)
            for(int j = 0; j < w; j++)
                digit[(xx + i) + W * (yy + j)] = 1.0f;
            input.down = true;
        }
        // Predict.
        else
        {
            if(input.down)
            {
                const float* const pred = xpredict(tinn, digit);
                dprint(pred, data.nops);
                const Index ix = ixmax(pred, data.nops);
                if(ix.val > 0.9f)
                    printf("%d\n", ix.i);
                else
                    printf("I do not recognize that digit\n");
                memset((void*) digit, 0, sizeof(digit));
            }
            input.down = false;
        }
        // Draw digit to screen.
        for(int x = 0; x < xres; x++)
        for(int y = 0; y < yres; y++)
        {
            const int xx = x / S;
            const int yy = y / S;
            digit[xx + W * yy] == 1.0f ?
                SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF):
                SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xFF);
            SDL_RenderDrawPoint(renderer, x, y);
        }
        SDL_RenderPresent(renderer);
        SDL_Delay(15);
    }
    #undef W
    #undef H
    #undef S
}

int main()
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    // Input and output size is harded coded here as machine learning
    // repositories usually don't include the input and output size in the data itself.
    const int nips = 256;
    const int nops = 10;
    // Hyper Parameters.
    // Learning rate is annealed and thus not constant.
    // It can be fine tuned along with the number of hidden layers.
    // Feel free to modify the anneal rate as well.
    const int nhid = 28;
    float rate = 1.0f;
    const float anneal = 0.99f;
    // Load the training set.
    const Data data = build("semeion.data", nips, nops);
    // Train, baby, train.
    const Tinn tinn = xtbuild(nips, nhid, nops);
    for(int i = 0; i < 200; i++)
    {
        shuffle(data);
        float error = 0.0f;
        for(int j = 0; j < data.rows; j++)
        {
            const float* const in = data.in[j];
            const float* const tg = data.tg[j];
            error += xttrain(tinn, in, tg, rate);
        }
        printf("error %.12f :: rate %f\n", (double) error / data.rows, (double) rate);
        rate *= anneal;
    }
    // This is how you save the neural network to disk.
    xtsave(tinn, "saved.tinn");
    xtfree(tinn);
    // This is how you load the neural network from disk.
    const Tinn loaded = xtload("saved.tinn");
    // Now we do a prediction with the neural network we loaded from disk.
    // SDL will create a window so that you can draw digits.
    // Enter the draw and predict loop:
    dploop(loaded, data);
    // All done. Let's clean up
    xtfree(loaded);
    dfree(data);
    return 0;
}
