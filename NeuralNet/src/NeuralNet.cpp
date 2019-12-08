//============================================================================
// Name        : NeuralNet.cpp
// Author      : abahansy
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <boost/algorithm/string.hpp>

// This reference is needed for typedef Layer, so it shall be kept here
class Neuron;

typedef std::vector<Neuron> Layer_t;


/*--------------- Class Neuron ----------------*/


class Neuron {
	struct Connection {
		double weight;
		double deltaWeight;
	};
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputValue(double);
	double getOutputValue(void) const;
	void feedForward(Layer_t &previousLayer);
	void calculateOutputGradients(double targetVal);
	void calculateHiddenGradients(const Layer_t &nextLayer);
	void updateInputWeights(Layer_t &previousLayer);
private:
	static double eta; //static member of the class, so all the instances can use it instead of putting as dynamic part of the class.
	static double alpha; // multiplier of the last weight change (momentum)
    static double randomWeight(void);
	static double activationFunc (double var);
	static double activationFuncDerivative (double var);
	double sumDOW(const Layer_t &nextLayer) const;

	unsigned m_myIndex;
	double outputValue;
	std::vector<Connection> outputWeights;
	double m_gradient;


};

double Neuron::eta = 0.15; // learning rate
double Neuron::alpha = 0.5; // momentum





Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {

	m_myIndex = myIndex;
	for (unsigned connectionNum = 0; connectionNum < numOutputs; ++connectionNum) {
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}
}

double Neuron::activationFunc (double var) {
	return tanh(var);
}

double Neuron::randomWeight(void) {

	return rand() / double(RAND_MAX);

}


double Neuron::activationFuncDerivative (double var) {
	return 1 - var*var;
}

void Neuron::setOutputValue(double value) {
	outputValue = value;
}

double Neuron::getOutputValue(void) const  {
	return outputValue;
}

void Neuron::feedForward(Layer_t &previousLayer) {
	double sum = 0.0;
	for(unsigned iter = 0; iter<previousLayer.size(); ++iter) {
		// add the result of multiplication of the output value of the neuron with the weighted value towards current neuron.
		sum += previousLayer[iter].outputValue * previousLayer[iter].outputWeights[m_myIndex].weight;
	}
	// calculate the output of the neuron
	outputValue = Neuron::activationFunc(sum);

}


void Neuron::calculateOutputGradients(double targetVal) {
	double delta = targetVal - outputValue;
	m_gradient = delta * Neuron::activationFuncDerivative(outputValue);

}

void Neuron::calculateHiddenGradients(const Layer_t &nextLayer) {
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::activationFuncDerivative(outputValue);
}

double Neuron::sumDOW(const Layer_t &nextLayer) const{
	double sum = 0.0;

	for (unsigned iter = 0; iter < nextLayer.size(); ++iter) {
		sum += outputWeights[iter].weight * nextLayer[iter].m_gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layer_t &previousLayer) {
	// The weights to be modified are in the connection container in the neurons of the previous layer
	for (unsigned iter = 0; iter < previousLayer.size(); ++iter) {
		Neuron &neuron = previousLayer[iter];
		double oldDeltaWeight = neuron.outputWeights[iter].deltaWeight;
		double newDeltaWeight =
				// Individual input, magnified by the gradient and train rate
				eta * neuron.getOutputValue() * m_gradient + alpha *oldDeltaWeight;
		neuron.outputWeights[iter].deltaWeight = newDeltaWeight;
		neuron.outputWeights[iter].weight += newDeltaWeight;
	}


}


/*--------------- Class Net ----------------*/

class Net {

public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputValues);
	void backProb(const std::vector<double> &targetValues);
	void getResults(std::vector<double> &resultsVal) const;
private:
	std::vector<Layer_t> m_layers;
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingError;

};

Net::Net(const std::vector<unsigned> &topology){
	m_error = 0.0;
	m_recentAverageError = 0;
	m_recentAverageSmoothingError = 0;
	unsigned numLayers = topology.size();

	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		// add new empty layer to the net
		m_layers.push_back(Layer_t());
		// define the number of outputs for neurons, use ternary operator to get the number of outputs.
		unsigned numOutputs = (layerNum == topology.size() - 1? 0: topology[layerNum+1]);
		// <= to add the bias neuron to the layer
		for (unsigned NeuronNum = 0; NeuronNum <= topology[layerNum]; ++NeuronNum) {
			// Construct the neuron with the number of outputs and its index in this layer
			m_layers.back().push_back(Neuron(numOutputs, NeuronNum));
			std::cout << "Added a new neuron ! :D " << std::endl;
		}

	}
}
void Net::feedForward(const std::vector<double> &inputValues) {
	// number of elements in inputValues is equal to the number of elements in the input layer
	assert(inputValues.size() == m_layers[0].size() - 1);

	// assign the input values to the input neurons
	for (unsigned i = 0; i < inputValues.size(); ++i) {
		m_layers[0][i].setOutputValue(inputValues[i]);
	}

	//do forward propagation
	// counter started with value 1 to skip the input layer
	for (unsigned iterLayer = 1; iterLayer < m_layers.size(); ++iterLayer) {
		//pointer to the previous later needed by the neuron to feed forward
		Layer_t &previousLayer = m_layers[iterLayer -1];
		// size() - 1 to skip the bias neuron
		for (unsigned iterNeuron = 0; iterNeuron < m_layers[iterLayer].size() -1; ++iterNeuron) {
			m_layers[iterLayer][iterNeuron].feedForward(previousLayer);
		}
	}


}

void Net::backProb(const std::vector<double> &targetValues) {

	Layer_t &outputLayer = m_layers.back();
	//update the error for the whole network
	m_error = 0;

	// Calculate the sum error
	for (unsigned i = 0; i < outputLayer.size(); ++i) {
		double delta = targetValues[i] - outputLayer[i].getOutputValue();
		m_error+= delta * delta;
	}

	m_error /= outputLayer.size(); //averaging the error
	m_error = sqrt(m_error); //square root of the error

	// implement the average measurement
	m_recentAverageError  = (m_recentAverageError * m_recentAverageSmoothingError + m_error) / (m_recentAverageSmoothingError +1.0);

	// calculate the output layer gradients
	// splitting the responsibilities  where net class will loop and neuron class will do the calculations of the gradient
	for (unsigned iter = 0; iter < outputLayer.size() -1 ; ++iter) { // skipping the bias neuron
		outputLayer[iter].calculateOutputGradients(targetValues[iter]);
	}

	//calculate the hidden layer gradients, will start calculation from the right to the one before the start
	for (unsigned iter = m_layers.size() - 2; iter > 0; --iter) {

		Layer_t &hiddenLayer = m_layers[iter]; // pointer the current layer
		Layer_t &nextLayer = m_layers[iter +1]; // pointer to the next layer

		for (int unsigned i = 0; i < hiddenLayer.size(); ++i) {
			hiddenLayer[i].calculateHiddenGradients(nextLayer);
		}

	}
	//update connection weights, loop over all layers except for the input and the output one
	for (unsigned i = m_layers.size()-1; i>0; ++i) {

		Layer_t &previousLayer = m_layers[i-1];
		Layer_t &layer = m_layers[i];

		for (unsigned j = 0; j < layer.size() - 1; ++j) {
		            layer[j].updateInputWeights(previousLayer);
		}
	}


}

void Net::getResults(std::vector<double> &resultsVal) const {

	resultsVal.clear();

	for (unsigned i = 0; i< m_layers.back().size() -1; ++i) {
		resultsVal.push_back(m_layers.back()[i].getOutputValue());
	}
}





int main() {
	// e.g., {4,2,1}
	std::vector<unsigned> topology;
	topology.push_back(4);
	topology.push_back(2);
	topology.push_back(1);

	Net my_net(topology);
//
//	const std::vector<double> inputValues;
//	my_net.feedForward(inputValues);
//
//	const std::vector<double> targetValues;
//	my_net.backProb(targetValues);
//
//	std::vector<double> resultsValues;
//	my_net.getResults(resultsValues);

	std::ifstream myFile ("/Users/abahnasy/Desktop/neural_net_in_cpp/NeuralNet/src/iris.csv");
//	std::cout << myFile.rdstate() << std::endl;

//	std::cout << " good()=" << myFile.good();
//	std::cout << " eof()=" << myFile.eof();
//	std::cout << " fail()=" << myFile.fail();
//	std::cout << " bad()=" << myFile.bad();

	static int counter = 0; //to skip the title row in csv file

	while(myFile.good()){

		std::string line;
		getline(myFile, line, '\n');
		if(counter++ != 0) {
			std::vector<double> sampleInputs;
			std::vector<double> targetValues;
			std::vector<std::string> results;
			std::string delims(",");
			boost::split(results, line, boost::is_any_of(delims));

			sampleInputs.push_back(std::stod(results.at(0)));
			sampleInputs.push_back(std::stod(results.at(1)));
			sampleInputs.push_back(std::stod(results.at(2)));
			sampleInputs.push_back(std::stod(results.at(3)));


			std::string targetValueName = results.at(4);
			if (targetValueName == "Setosa") {
				targetValues.push_back(0);
			} else if (targetValueName == "Versicolor") {
				targetValues.push_back(1);
			} else {/*Virginica*/
				targetValues.push_back(2);
			}
		}
		//counter++;



	}

	std::cout << counter << "\n";




	return 0;
}
