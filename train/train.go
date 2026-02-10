package train

import (
	"allenbercero/digit_recognizer/ml"
	"math/rand"
)

// Neuron initializer. neuron count and parameter count is a slice
// because it can be different in each hidden layer
func InitNeuron(neuronCount int, paramCount int) []ml.Neuron {
	var neurons = make([]ml.Neuron, neuronCount)

	for i := range neurons {
		neurons[i].Bias = rand.Float64()
		neurons[i].Weights = make([]float64, paramCount)
		for weight := range neurons[i].Weights {
			neurons[i].Weights[weight] = rand.Float64()
		}
	}

	return neurons
}
