//
// GENANN - Minimal C Artificial Neural Network
//
// Copyright (c) 2015, 2016 Lewis Van Winkle
//
// http://CodePlea.com
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgement in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//
//
// This software has been altered from its original state. Namely white space edits
// and formatting but most importantly the library has been moved into a single
// static inline header file.
//
// - Gustav Louw 2018

#pragma once

#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double (*genann_actfun)(double a);

typedef struct
{
    // How many inputs, outputs, and hidden neurons.
    int inputs;
    int hidden_layers;
    int hidden;
    int outputs;
    // Which activation function to use for hidden neurons. Default: gennann_act_sigmoid_cached.
    genann_actfun activation_hidden;
    // Which activation function to use for output. Default: gennann_act_sigmoid_cached.
    genann_actfun activation_output;
    // Total number of weights, and size of weights buffer.
    int total_weights;
    // Total number of neurons + inputs and size of output buffer.
    int total_neurons;
    // All weights (total_weights long).
    double *weight;
    // Stores input array and output of each neuron (total_neurons long).
    double *output;
    // Stores delta of each hidden and output neuron (total_neurons - inputs long).
    double *delta;

}
Genann;

static inline double genann_act_sigmoid(double a)
{
    return a < -45.0 ? 0 : a > 45.0 ? 1.0 : 1.0 / (1 + exp(-a));
}

static inline double genann_act_sigmoid_cached(double a)
{
    // If you're optimizing for memory usage, just
    // delete this entire function and replace references
    // of genann_act_sigmoid_cached to genann_act_sigmoid.
    const double min = -15.0;
    const double max = 15.0;
    static double interval;
    static int initialized = 0;
    static double lookup[4096];
    const int lookup_size = sizeof(lookup) / sizeof(*lookup);
    // Calculate entire lookup table on first run.
    if(!initialized)
    {
        interval = (max - min) / lookup_size;
        for(int i = 0; i < lookup_size; ++i)
            lookup[i] = genann_act_sigmoid(min + interval * i);
        // This is down here to make this thread safe.
        initialized = 1;
    }
    const int i = (int) ((a - min) / interval + 0.5);
    return i <= 0 ? lookup[0] : i >= lookup_size ? lookup[lookup_size - 1] : lookup[i];
}

static inline double genann_act_threshold(double a)
{
    return a > 0;
}

static inline double genann_act_linear(double a)
{
    return a;
}

// We use the following for uniform random numbers between 0 and 1.
// If you have a better function, redefine this macro.
static inline double genann_random()
{
    return (double) rand() / RAND_MAX;
}

static inline void genann_randomize(Genann *ann)
{
    for(int i = 0; i < ann->total_weights; ++i)
    {
        double r = genann_random();
        // Sets weights from -0.5 to 0.5.
        ann->weight[i] = r - 0.5;
    }
}

static inline Genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs)
{
    if(hidden_layers < 0)
        return 0;
    if(inputs < 1)
        return 0;
    if(outputs < 1)
        return 0;
    if(hidden_layers > 0 && hidden < 1)
        return 0;
    const int hidden_weights = hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
    const int total_weights = hidden_weights + output_weights;
    const int total_neurons = inputs + hidden * hidden_layers + outputs;
    // Allocate extra size for weights, outputs, and deltas.
    const int size = sizeof(Genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    Genann *ret = malloc(size);
    if(!ret)
        return 0;
    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;
    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;
    // Set pointers.
    ret->weight = (double*)((char*)ret + sizeof(Genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;
    genann_randomize(ret);
    ret->activation_hidden = genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid_cached;
    return ret;
}

static inline void genann_free(Genann *ann)
{
    // The weight, output, and delta pointers go to the same buffer.
    free(ann);
}

static inline Genann *genann_read(FILE *in)
{
    int inputs, hidden_layers, hidden, outputs;
    errno = 0;
    int rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if(rc < 4 || errno != 0)
    {
        perror("fscanf");
        return NULL;
    }
    Genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);
    for(int i = 0; i < ann->total_weights; ++i)
    {
        errno = 0;
        rc = fscanf(in, " %le", ann->weight + i);
        if(rc < 1 || errno != 0)
        {
            perror("fscanf");
            genann_free(ann);
            return NULL;
        }
    }
    return ann;
}

static inline Genann *genann_copy(Genann const *ann)
{
    const int size = sizeof(Genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    Genann *ret = malloc(size);
    if(!ret)
        return 0;
    memcpy(ret, ann, size);
    // Set pointers.
    ret->weight = (double*)((char*)ret + sizeof(Genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;
    return ret;
}

static inline double const *genann_run(Genann const *ann, double const *inputs)
{
    double const *w = ann->weight;
    double *o = ann->output + ann->inputs;
    double const *i = ann->output;
    // Copy the inputs to the scratch area, where we also store each neuron's
    // output, for consistency. This way the first layer isn't a special case.
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);
    const genann_actfun act = ann->activation_hidden;
    const genann_actfun acto = ann->activation_output;
    // Figure hidden layers, if any.
    for(int h = 0; h < ann->hidden_layers; ++h)
    {
        for(int j = 0; j < ann->hidden; ++j)
        {
            double sum = *w++ * -1.0;
            for(int k = 0; k < (h == 0 ? ann->inputs : ann->hidden); ++k)
            {
                sum += *w++ * i[k];
            }
            *o++ = act(sum);
        }
        i += (h == 0 ? ann->inputs : ann->hidden);
    }
    double const *ret = o;
    // Figure output layer.
    for(int j = 0; j < ann->outputs; ++j)
    {
        double sum = *w++ * -1.0;
        for(int k = 0; k < (ann->hidden_layers ? ann->hidden : ann->inputs); ++k)
            sum += *w++ * i[k];
        *o++ = acto(sum);
    }
    // Sanity check that we used all weights and wrote all outputs.
    assert(w - ann->weight == ann->total_weights);
    assert(o - ann->output == ann->total_neurons);
    return ret;
}

static inline void genann_train(Genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
    // To begin with, we must run the network forward.
    genann_run(ann, inputs);
    // First set the output layer deltas.
    {
        // First output.
        double const *o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers;
        // First delta.
        double *d = ann->delta + ann->hidden * ann->hidden_layers;
        // First desired output.
        double const *t = desired_outputs;
        // Set output layer deltas.
        if(ann->activation_output == genann_act_linear)
        {
            for(int j = 0; j < ann->outputs; ++j)
            {
                *d++ = *t++ - *o++;
            }
        }
        else
        {
            for(int j = 0; j < ann->outputs; ++j)
            {
                *d++ = (*t - *o) * *o * (1.0 - *o);
                ++o; ++t;
            }
        }
    }
    // Set hidden layer deltas, start on last layer and work backwards.
    // Note that loop is skipped in the case of hidden_layers == 0.
    for(int h = ann->hidden_layers - 1; h >= 0; --h)
    {
        // Find first output and delta in this layer.
        double const *o = ann->output + ann->inputs + (h * ann->hidden);
        double *d = ann->delta + (h * ann->hidden);
        // Find first delta in following layer (which may be hidden or output).
        double const * const dd = ann->delta + ((h+1) * ann->hidden);
        // Find first weight in following layer (which may be hidden or output).
        double const * const ww = ann->weight + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));
        for(int j = 0; j < ann->hidden; ++j)
        {
            double delta = 0;
            for(int k = 0; k < (h == ann->hidden_layers-1 ? ann->outputs : ann->hidden); ++k)
            {
                const double forward_delta = dd[k];
                const int windex = k * (ann->hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
            }
            *d = *o * (1.0-*o) * delta;
            ++d; ++o;
        }
    }
    // Train the outputs.
    {
        // Find first output delta.
        // First output delta.
        double const *d = ann->delta + ann->hidden * ann->hidden_layers;
        // Find first weight to first output delta.
        double *w = ann->weight + (ann->hidden_layers
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * ann->hidden * (ann->hidden_layers-1))
                : (0));
        // Find first output in previous layer.
        double const * const i = ann->output + (ann->hidden_layers
                ? (ann->inputs + (ann->hidden) * (ann->hidden_layers-1))
                : 0);
        // Set output layer weights.
        for(int j = 0; j < ann->outputs; ++j) {
            for(int k = 0; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
                if(k == 0)
                {
                    *w++ += *d * learning_rate * -1.0;
                }
                else
                {
                    *w++ += *d * learning_rate * i[k-1];
                }
            }
            ++d;
        }
        assert(w - ann->weight == ann->total_weights);
    }
    // Train the hidden layers.
    for(int h = ann->hidden_layers - 1; h >= 0; --h)
    {
        // Find first delta in this layer.
        double const *d = ann->delta + (h * ann->hidden);
        // Find first input to this layer.
        double const *i = ann->output + (h
                ? (ann->inputs + ann->hidden * (h-1))
                : 0);
        // Find first weight to this layer.
        double *w = ann->weight + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);
        for(int j = 0; j < ann->hidden; ++j)
        {
            for(int k = 0; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k)
            {
                if (k == 0)
                {
                    *w++ += *d * learning_rate * -1.0;
                }
                else
                {
                    *w++ += *d * learning_rate * i[k-1];
                }
            }
            ++d;
        }
    }
}

static inline void genann_write(Genann const *ann, FILE *out)
{
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);
    for(int i = 0; i < ann->total_weights; ++i)
        fprintf(out, " %.20e", ann->weight[i]);
}
