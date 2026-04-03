package ml

import (
	"errors"
	"math/rand/v2"
)

type Layer struct {
	Neurons [][]float64
	Bias    []float64
	// WHERE len(Bias) == number of Neurons/rows
	// Bias[0] maps to Neurons[0][]
}

type Output struct {
	PreviousLayer *Layer
	RawLogits     [10]float64
	Softmaxed     [10]float64
	CurrentLayer  Layer
}

/*
[4][5]float64 is row x colum

[
	[1, 2, 3, 4, 5], row 1
	[1, 2, 3, 4, 5], row 2
	[1, 2, 3, 4, 5], row 3
	[1, 2, 3, 4, 5], row 4
]

- in our neural network a row is a neuron
- biases should be mapped for each neuron
- len(Biases) == num of Weights rows
*/

// -- HIDDEN LAYER --
// Hidden Layer Constructor
func NewHiddenLayer(neuronCount int, weightsPerNeuron int) (layer Layer, err error) {
	if neuronCount < 1 || weightsPerNeuron < 1 {
		err = errors.New("Arg neuronCount and weightsPerNeuron must at least be 1")
		return
	}

	layer.Neurons = CreateMatrix(neuronCount, weightsPerNeuron)
	layer.Bias = make([]float64, neuronCount)
	return
}

// Used for model training only, initializes the output layer with ranodm weights
func (layer *Layer) Init() {
	for neuron := range layer.Neurons {
		for weight := range layer.Neurons[neuron] {
			layer.Neurons[neuron][weight] = rand.Float64()
		}
	}

	for bias := range layer.Bias {
		layer.Bias[bias] = rand.Float64()
	}
}

// Inference
func (layer *Layer) Run() {}

// -- OUTPUT LAYER --
// Output Layer Constructor
func NewOutputLayer(prevLayer *Layer) (output Output, err error) {
	// previous layer's neuron count = output layer's weight count
	numOfWeights := len(prevLayer.Neurons)

	if numOfWeights < 1 {
		err = errors.New("Previous layer's neuron count should be at least 1")
		return
	}

	output.PreviousLayer = prevLayer
	output.CurrentLayer.Neurons = CreateMatrix(10, numOfWeights)
	output.CurrentLayer.Bias = make([]float64, 10)

	return
}

// Used for model training only, initializes the output layer with ranodm weights
func (output *Output) Init() {
	output.CurrentLayer.Init()
}

// Inference
func (output *Output) Run() {
	CreateMatrix(len(output.PreviousLayer.Neurons), len(output.CurrentLayer.Neurons))
}
