// This program uses a modified version of the Genann Neural Network Library
// to learn hand written digits.
//
// Get the training data from the machine learning database:
//     wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data

#include <errno.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define toss(t, n) ((t*) malloc((n) * sizeof(t)))

#define retoss(ptr, t, n) (ptr = (t*) realloc((ptr), (n) * sizeof(t)))

typedef double (*genann_actfun)(double);

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

typedef struct
{
    int inputs;
    int hidden_layers;
    int hidden;
    int outputs;
    genann_actfun activation_hidden;
    genann_actfun activation_output;
    int total_weights;
    int total_neurons;
    double* weight;
    double* output;
    double* delta;

}
Genann;

static double genann_act_sigmoid(const double a)
{
    return a < -45.0 ? 0 : a > 45.0 ? 1.0 : 1.0 / (1 + exp(-a));
}

static void genann_randomize(Genann* const ann)
{
    for(int i = 0; i < ann->total_weights; i++)
    {
        double r = (double) rand() / RAND_MAX;
        ann->weight[i] = r - 0.5;
    }
}

// Clean this up. The mallocs do not look right.
static Genann *genann_init(const int inputs, const int hidden_layers, const int hidden, const int outputs)
{
    const int hidden_weights = hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
    const int total_weights = hidden_weights + output_weights;
    const int total_neurons = inputs + hidden * hidden_layers + outputs;
    // Allocate extra size for weights, outputs, and deltas.
    const int size = sizeof(Genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    Genann* ret = (Genann*) malloc(size);
    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;
    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;
    // Set pointers.
    ret->weight = (double*) ((char*) ret + sizeof(Genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;
    ret->activation_hidden = genann_act_sigmoid;
    ret->activation_output = genann_act_sigmoid;
    genann_randomize(ret);
    return ret;
}

static double const *genann_run(Genann const *ann, double const *inputs)
{
    const double* w = ann->weight;
    double* o = ann->output + ann->inputs;
    const double* i = ann->output;
    // Copy the inputs to the scratch area, where we also store each neuron's
    // output, for consistency. This way the first layer isn't a special case.
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);
    const genann_actfun act = ann->activation_hidden;
    const genann_actfun acto = ann->activation_output;
    // Figure hidden layers, if any.
    for(int h = 0; h < ann->hidden_layers; h++)
    {
        for(int j = 0; j < ann->hidden; j++)
        {
            double sum = *w++ * -1.0;
            for(int k = 0; k < (h == 0 ? ann->inputs : ann->hidden); k++)
                sum += *w++ * i[k];
            *o++ = act(sum);
        }
        i += (h == 0 ? ann->inputs : ann->hidden);
    }
    const double* ret = o;
    // Figure output layer.
    for(int j = 0; j < ann->outputs; ++j)
    {
        double sum = *w++ * -1.0;
        for(int k = 0; k < (ann->hidden_layers ? ann->hidden : ann->inputs); ++k)
            sum += *w++ * i[k];
        *o++ = acto(sum);
    }
    return ret;
}

static void genann_train(const Genann* ann, const double* inputs, const double* desired_outputs, const double rate)
{
    // To begin with, we must run the network forward.
    genann_run(ann, inputs);
    // First set the output layer deltas.
    {
        // First output.
        const double* o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers;
        // First delta.
        double* d = ann->delta + ann->hidden * ann->hidden_layers;
        // First desired output.
        const double* t = desired_outputs;
        // Set output layer deltas.
        for(int j = 0; j < ann->outputs; j++, o++, t++)
            *d++ = (*t - *o) * *o * (1.0 - *o);
    }
    // Set hidden layer deltas, start on last layer and work backwards.
    // Note that loop is skipped in the case of hidden_layers == 0.
    for(int h = ann->hidden_layers - 1; h >= 0; h--)
    {
        // Find first output and delta in this layer.
        const double* o = ann->output + ann->inputs + (h * ann->hidden);
        double* d = ann->delta + (h * ann->hidden);
        // Find first delta in following layer (which may be hidden or output).
        const double* const dd = ann->delta + ((h + 1) * ann->hidden);
        // Find first weight in following layer (which may be hidden or output).
        const double* const ww = ann->weight + ((ann->inputs + 1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));
        for(int j = 0; j < ann->hidden; j++, d++, o++)
        {
            double delta = 0;
            for(int k = 0; k < (h == ann->hidden_layers - 1 ? ann->outputs : ann->hidden); k++)
            {
                const double forward_delta = dd[k];
                const int windex = k * (ann->hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
            }
            *d = *o * (1.0 - *o) * delta;
        }
    }
    // Train the outputs.
    {
        // Find first output delta. First output delta.
        const double* d = ann->delta + ann->hidden * ann->hidden_layers;
        // Find first weight to first output delta.
        double* w = ann->weight + (ann->hidden_layers ? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * ann->hidden * (ann->hidden_layers - 1)) : 0);
        // Find first output in previous layer.
        const double* const i = ann->output + (ann->hidden_layers ? (ann->inputs + ann->hidden * (ann->hidden_layers - 1)) : 0);
        // Set output layer weights.
        for(int j = 0; j < ann->outputs; ++j, ++d)
            for(int k = 0; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; k++)
                *w++ += (k == 0) ? (*d * rate * -1.0) : (*d * rate * i[k - 1]);
    }
    // Train the hidden layers.
    for(int h = ann->hidden_layers - 1; h >= 0; h--)
    {
        // Find first delta in this layer.
        const double* d = ann->delta + (h * ann->hidden);
        // Find first input to this layer.
        double* const i = ann->output + (h ? (ann->inputs + ann->hidden * (h - 1)) : 0);
        // Find first weight to this layer.
        double* w = ann->weight + (h ? ((ann->inputs + 1) * ann->hidden + (ann->hidden + 1) * (ann->hidden) * (h - 1)) : 0);
        for(int j = 0; j < ann->hidden; j++, d++)
            for(int k = 0; k < (h == 0 ? ann->inputs : ann->hidden) + 1; k++)
                *w++ += (k == 0) ? (*d * rate * -1.0) : (*d * rate * i[k - 1]);
    }
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

static void print(const double* const arr, const int size, const double thresh)
{
    for(int i = 0; i < size; i++)
        printf("%d ", arr[i] > thresh);
}

static int cmp(const double* const a, const double* const b, const int size, const double thresh)
{
    for(int i = 0; i < size; i++)
    {
        const int aa = a[i] > thresh;
        const int bb = b[i] > thresh;
        if(aa != bb)
            return 0;
    }
    return 1;
}

static void predict(Genann* ann, const Data d)
{
    const double thresh = 0.8;
    int matches = 0;
    for(int i = d.split; i < d.rows; i++)
    {
        // Prediciton.
        const double* const pred = genann_run(ann, d.id[i]);
        const double* const real = d.od[i];
        print(pred, d.ocols, thresh);
        printf(":: ");
        print(real, d.ocols, thresh);
        const int match = cmp(pred, real, d.ocols, thresh);
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
    free(ann);
    dfree(data);
    return 0;
}
