#pragma once

#include <vector>
#include <string>

using std::vector;
using std::string;

struct TinnState;

class Tinn {
public:
	Tinn(int n_inputs, int n_hidden, int n_outputs);
	Tinn(string path);

	double train(vector<double> input, vector<double> target, double rate);
	vector<double> predict(vector<double> input);

	void save(string path);
private:
	TinnState forward_propogate(vector<double> input);
	void back_propogate(TinnState state, vector<double> input, vector<double> target, double rate);

	double get_input_weight(int hidden, int input) const {
		return weights[hidden * n_inputs + input];
	}
	double get_hidden_weight(int output, int hidden) const {
		return weights[output * n_hidden + hidden + n_hidden * n_inputs];
	}
	void set_input_weight(int hidden, int input, double weight) {
		weights[hidden * n_inputs + input] = weight;
	}
	void set_hidden_weight(int output, int hidden, double weight) {
		weights[output * n_hidden + hidden + n_hidden * n_inputs] = weight;
	}

	void randomize_weights_biases();

	int n_inputs, n_hidden, n_outputs;
	const static int n_biases = 2;

	vector<double> weights, biases;
};

struct TinnState {
public:
	TinnState(vector<double> hidden, vector<double> output) : hidden{hidden}, output{output} {}
	double get_hidden(int n) const { return hidden[n]; }
	double get_output(int n) const { return output[n]; }
	vector<double> get_outputs() const { return output; }
private:
	vector<double> hidden, output;
};

double activation(double x);
double partial_activation(double x);
double error(double target, double output);
double partial_error(double target, double output);
double total_error(vector<double> target, vector<double> output);
