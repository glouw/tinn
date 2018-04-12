#include "Tinn.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <random>

using std::vector;
using std::string;

struct Data {
public:
	Data(string path, int n_inputs, int n_outputs);

	void shuffle();

	int get_n_rows() const { return n_rows; }
	vector<double> get_input(int input) const { return inputs[input]; }
	vector<double> get_target(int target) const { return targets[target]; }

private:
	void parse(string line, int row);

	vector<vector<double>> inputs, targets;
	int n_inputs, n_outputs, n_rows;
};

class Line {
public:
	friend std::istream &operator>>(std::istream &is, Line &l) {
		std::getline(is, l.data);
		return is;
	}
private:
	std::string data;
};

Data::Data(string path, int n_inputs, int n_outputs) :
	n_inputs{n_inputs},
	n_outputs{n_outputs} {
	std::ifstream file(path);
	std::ifstream counter(path);
	n_rows = std::count(std::istreambuf_iterator<char>(counter), std::istreambuf_iterator<char>(), '\n');

	vector<vector<double>> prototype(n_rows);
	inputs = prototype;
	targets = prototype;

	for(int row = 0; row < n_rows; row++) {
		string line;
		std::getline(file, line);
		parse(line, row);
	}
}

void Data::parse(string line, int row) {
	int n_columns = n_inputs + n_outputs;
	string delimiter{" "};
	vector<double> values;
	values.reserve(n_columns);

	inputs[row].reserve(n_inputs);
	targets[row].reserve(n_outputs);

	size_t pos = 0;
	std::string token;

	while ((pos = line.find(delimiter)) != std::string::npos) {
   		token = line.substr(0, pos);
			values.push_back(std::stod(token));
	    line.erase(0, pos + delimiter.length());
	}

	for(int column = 0; column < n_columns; column++) {
		if(column < n_inputs)
			inputs.at(row).push_back(values[column]);
		else
			targets.at(row).push_back(values[column]);
	}
}

void Data::shuffle() {
	std::uniform_int_distribution<int> distribution{0, n_rows - 1};
	std::default_random_engine generator{std::chrono::system_clock::now().time_since_epoch().count()};
	// std::default_random_engine generator{0};

	for(int row = 0; row < n_rows; row++) {
		int swap = distribution(generator);
		std::iter_swap(inputs.begin() + row, inputs.begin() + swap);
		std::iter_swap(targets.begin() + row, targets.begin() + swap);
	}
}

int main() {
	const int n_inputs{256};
	const int n_outputs{10};

	const int n_hidden{28};
	double rate{1.0f};
	const double anneal{0.99f};

	Data data{"semeion.data", n_inputs, n_outputs};

	Tinn tinn{n_inputs, n_hidden, n_outputs};

	int rounds;
	std::cout << "How many rounds of training?" << std::endl;
	std::cin >> rounds;

	std::cout.precision(4);

	for(int i = 0; i < rounds; i++) {
		data.shuffle();

		double error{0};

		for(int j = 0; j < data.get_n_rows(); j++) {
			vector<double> input = data.get_input(j);
			vector<double> target = data.get_target(j);

			error += tinn.train(input, target, rate);
		}

		std::cout << "error: " << std::fixed << error / data.get_n_rows() << " learning rate: " << rate << std::endl;
		rate *= anneal;
	}

	tinn.save("saved.tinn");
	Tinn loaded{"saved.tinn"};

	vector<double> input = data.get_input(0);
	vector<double> target = data.get_target(0);
	vector<double> prediction = tinn.predict(input);

	for(auto i = target.begin(); i != target.end(); i++) { std::cout << std::fixed << *i << " "; } std::cout << std::endl;
	for(auto i = prediction.begin(); i != prediction.end(); i++) { std::cout << std::fixed << *i << " "; } std::cout << std::endl;
}
