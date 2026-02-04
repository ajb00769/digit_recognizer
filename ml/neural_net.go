package ml

import (
	"math"
	"math/rand"
)

// Input Layer 28 * 28 matrix flatten. We do not mutate the Input so
// it's safe to pass in the Input parameter value itself instead of
// creating a copy of it
func Input(matrixInput *[28][28]float64) [784]float64 {
	// NOTE: using fixed size array to enforce rigid input lengths
	// anything less or greater than the fixed size should not be
	// acceptable since the feed-forward mechanism to other layers
	// will need this fixed sizing
	var flattened [784]float64 = [784]float64{}

	for row := range matrixInput {
		for item := range matrixInput[row] {
			// hardcoded 28 in the formula since we expect 28 to be
			// a constant size
			flattened[(row*28)+item] = matrixInput[row][item]
		}
	}

	return flattened
}

// TODO: Hidden Layer
func LayerOne() {
	// Implement hidden layer logic here
}

func LayerTwo() {
	// Implement hidden layer logic here
}

// TODO: Output Layer
func Output() {
	// Implement output layer logic here
	// Use softmax activation function
}

// TODO: Activation Functions
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Used in the output layer
func Softmax() {}

// Neuron initializer. neron count and parameter count is a slice
// because it can be different in each hidden layer
func InitNeuron(neuronCount int, paramCount int) (neurons [][]float64, biases []float64) {
	neurons = make([][]float64, neuronCount)
	biases = make([]float64, neuronCount)

	for neuron := range neurons {
		neurons[neuron] = make([]float64, paramCount)
		for param := range neurons[neuron] {
			neurons[neuron][param] = rand.Float64()
		}
		biases[neuron] = rand.Float64()
	}

	return neurons, biases
}
