package ml

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
)

type Neuron struct {
	Weights []float64
}

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

type HiddenLayer struct {
	LayerNum uint
	Neurons  []Neuron
	Bias     float64
}

func NewHiddenLayer(layerNum uint, neuronCount uint, paramsPerNeuron uint) (layer HiddenLayer, err error) {
	if neuronCount == 0 || paramsPerNeuron == 0 {
		err = errors.New("Arg neuronCount and paramsPerNeuron must at least be 1")
		return
	}

	layer = HiddenLayer{layerNum, make([]Neuron, neuronCount), rand.Float64()}

	for i := range layer.Neurons {
		layer.Neurons[i].Weights = make([]float64, paramsPerNeuron)
	}
	return
}

func (hl *HiddenLayer) LoadWeights() {
	// check if weights file exist
	// check if weights file is not empty
	// check if weights for this layer match the hyperparams in CreateHiddenLayer
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

func ForwardPropagate(matrixInput *[28][28]float64, hiddenLayers []*HiddenLayer) error {
	flattened := Input(matrixInput)

	// input layer to first hidden layer
	inputMatrix := make([][]float64, 1)
	inputMatrix[0] = flattened[:] // convert slice to matrix

	result, err := MatMul(inputMatrix, neuronToMatrix(hiddenLayers[0].Neurons))

	if err != nil {
		return err
	}

	activatedResult := CreateMatrix(len(result), len(result[0]))

	// add bias to each resulting weight from matmul and apply sigmoid/activation function
	for row := range result {
		for col := range result[row] {
			activated := Sigmoid(result[row][col] + hiddenLayers[0].Bias)
			activatedResult[row][col] = activated
		}
	}

	// for loop across all hidden layers
	for layer := range hiddenLayers {
		if layer == 0 {
			continue // skip first layer already performed above
		}

		result, err := MatMul(activatedResult, neuronToMatrix(hiddenLayers[layer].Neurons))

		if err != nil {
			return err
		}

		activatedResult = CreateMatrix(len(result), len(result[0]))

		for row := range result {
			for col := range result[row] {
				activated := Sigmoid(result[row][col] + hiddenLayers[layer].Bias)
				activatedResult[row][col] = activated
			}
		}
	}

	return nil
}

// Transpose function to convert []Neuron into a column-based [][]float64
// We treat 1 Neuron as a column of a matrix because in Matrix Multiplication
// the order of operations is to multiply each row of the left matrix against
// each column on the right matrix, and then sum their products. Since []Neuron
// is being stored as rows, we need to convert it into a column
func neuronToMatrix(n []Neuron) [][]float64 {
	rows := len(n[0].Weights)
	matrix := CreateMatrix(rows, len(n))

	for row := range rows {
		for col := range len(n) {
			matrix[row][col] = n[col].Weights[row]
		}
	}

	return matrix
}

func sliceToMatrix(s []float64) [][]float64 {
	rows := len(s)
	matrix := CreateMatrix(rows, 1)

	for row := range rows {
		matrix[row] = s
	}

	return matrix
}
