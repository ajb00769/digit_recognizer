package ml

import (
	"math"
	"slices"
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

// struct
type hiddenLayer struct {
	layerOrderId int
	neuronCount  int
}

// Hidden Layer, accepts a slice of hiddenLayer structs where you can set
// the number of neurons per layer and define the order of execution per layer
func HiddenLayer(layerStruct []hiddenLayer) {
	// Implement hidden layer logic here
}

// TODO: Output Layer
func Output() {
	// Implement output layer logic here
	// Use softmax activation function
}

// Activation Functions
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Activation function used in the output layer
func Softmax(logits []float64) []float64 {
	// get max value in slice for exponent rule
	max := slices.Max(logits)

	var total float64 = 0.0
	output := make([]float64, len(logits))

	for i := range len(logits) {
		// populate the output slice with all numerators to be divided by the total
		output[i] = math.Exp(logits[i] - max)
		total += output[i]
	}

	// perform softmax exponentiated numerator divided by total and replace
	// the numerator with the final value
	for i := range len(output) {
		output[i] /= total
	}

	return output
}

// The final output should be the index of the result of slices.Max()
// against the returned slice of the Softmax() function
type Neuron struct {
	Weights []float64
	Bias    float64
}

// TODO: load neuron function used during inference
func LoadNeuron() {}
