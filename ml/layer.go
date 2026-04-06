package ml

import (
	"errors"
	"math"
	"math/rand/v2"
)

type NetworkLayer interface {
	Init()
	Run()
}

type Layer struct {
	Neurons   [][]float64
	Bias      []float64
	Activated [][]float64
	// WHERE len(Bias) == number of Neurons/rows
	// Bias[0] maps to Neurons[0][]
}

type Output struct {
	RawLogits    [10]float64
	Softmaxed    [10]float64
	CurrentLayer Layer
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
	layer.Bias = make([]float64, weightsPerNeuron)
	return
}

// Used for model training only, initializes the output layer with ranodm weights
func (layer *Layer) Init() {
	for neuron := range layer.Neurons {
		for weight := range layer.Neurons[neuron] {
			layer.Neurons[neuron][weight] = xavierInitRandom()
		}
	}

	for bias := range layer.Bias {
		layer.Bias[bias] = xavierInitRandom()
	}
}

// TODO
// Inference
func (layer *Layer) Forward(prevOutput []float64) (activated []float64, err error) {
	matrix := CreateMatrix(1, len(prevOutput))
	matrix[0] = prevOutput

	result, err := MatMul(matrix, layer.Neurons)

	if err != nil {
		return nil, err
	}

	activated = make([]float64, len(result[0]))

	if err != nil {
		return nil, err
	}

	for i := range result[0] {
		tmp := result[0][i] + layer.Bias[i]
		activated[i] = Sigmoid(tmp)
	}

	return activated, nil
}

// -- OUTPUT LAYER --
// Output Layer Constructor
func NewOutputLayer(prevLayer *Layer) (output Output, err error) {
	// previous layer's neuron count = output layer's weight count
	numOfWeights := len(prevLayer.Neurons[0])

	if numOfWeights < 1 {
		err = errors.New("Previous layer's neuron count should be at least 1")
		return
	}

	output.CurrentLayer.Neurons = CreateMatrix(numOfWeights, 10)
	output.CurrentLayer.Bias = make([]float64, 10)

	return
}

// Used for model training only, initializes the output layer with ranodm weights
func (output *Output) Init() {
	output.CurrentLayer.Init()
}

// TODO
// Inference
func (output *Output) Forward(previousOutput []float64) ([]float64, error) {
	result, err := output.CurrentLayer.Forward(previousOutput)

	if err != nil {
		return nil, err
	}

	return Softmax(result), nil
}

func xavierInitRandom() float64 {
	x := math.Sqrt(6.0 / (784.0 + 10.0))
	return rand.Float64()*2*x - x
}
