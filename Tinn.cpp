#include "Tinn.h"

#include <random>
#include <chrono>
#include <cmath>
#include <fstream>

Tinn::Tinn(int n_inputs, int n_hidden, int n_outputs) :
	n_inputs{n_inputs},
	n_hidden{n_hidden},
	n_outputs{n_outputs} {
	weights.reserve((n_inputs + n_outputs) * n_hidden);
	biases.reserve(2);
	randomize_weights_biases();
}

Tinn::Tinn(std::string path) {
	std::ifstream file{path};
	file >> n_inputs >> n_hidden >> n_outputs;
	
	weights.reserve((n_inputs + n_outputs) * n_hidden);
	biases.reserve(2);

	for(int i = 0; i < n_biases; i++) {
		double temp;
		file >> temp;
		biases.push_back(temp);
	}
	for(int i = 0; i < (n_inputs + n_outputs) * n_hidden; i++) {
		double temp;
		file >> temp;
		weights.push_back(temp);
	}
}

double Tinn::train(vector<double> input, vector<double> target, double rate) {
	TinnState state = forward_propogate(input);
	back_propogate(state, input, target, rate);
	return total_error(target, state.get_outputs());
}

vector<double> Tinn::predict(vector<double> input) {
	return forward_propogate(input).get_outputs();
}

void Tinn::save(std::string path) {
	std::ofstream file{path};

	file << n_inputs << " " << n_hidden << " " << n_outputs << std::endl;

	for(int i = 0; i < n_biases; i++) file << biases[i] << std::endl;
	for(int i = 0; i < (n_inputs + n_outputs) * n_hidden; i++) file << weights[i] << std::endl;
}

void Tinn::randomize_weights_biases() {
	std::uniform_real_distribution<double> distribution{-0.5, 0.5};
	std::default_random_engine generator{std::chrono::system_clock::now().time_since_epoch().count()};

	for(double& w : weights) w = distribution(generator);
	for(double& b : biases) b = distribution(generator);
}

TinnState Tinn::forward_propogate(vector<double> input) {
	vector<double> hidden(n_hidden);
	vector<double> output(n_outputs);

	for(int i = 0; i < n_hidden; i++) {
		double sum{0};

		for(int j = 0; j < n_inputs; j++) {
			sum += input[j] * get_input_weight(i, j);
		}
		
		hidden[i] = activation(sum + biases[0]);
	}

	for(int i = 0; i < n_outputs; i++) {
		double sum{0};

		for(int j = 0; j < n_hidden; j++) {
			sum+= hidden[j] * get_hidden_weight(i, j);
		}

		output[i] = activation(sum + biases[1]);
	}

	TinnState state{hidden, output};
	return state;
}

void Tinn::back_propogate(TinnState state, vector<double> input, vector<double> target, double rate) {
	for(int i = 0; i < n_hidden; i++) {
		double sum{0};

		for(int j = 0; j < n_outputs; j++) {
			double a{partial_error(state.get_output(j), target[j])};
			double b{partial_activation(state.get_output(j))};

			sum += a * b * get_hidden_weight(j, i);
			set_hidden_weight(j, i, get_hidden_weight(j, i) - rate * a * b * state.get_hidden(i));
		}

		for(int j = 0; j < n_inputs; j++) {
			double delta = rate * sum * partial_activation(state.get_hidden(i)) * input[j];
			set_input_weight(i, j, get_input_weight(i, j) - delta);
		}
	}
}

double activation(double x) {
	return 1.0f / (1.0f + exp(-x));
}

double partial_activation(double x) {
	return x * (1.0f - x);
}

double error(double target, double output) {
	return 0.5f * pow(target - output, 2);
}

double partial_error(double target, double output) {
	return target - output;
}

double total_error(vector<double> target, vector<double> output) {
	double sum{0};
	for(int i = 0; i < target.size(); i++)
		sum += error(target[i], output[i]);
	return sum;
}

