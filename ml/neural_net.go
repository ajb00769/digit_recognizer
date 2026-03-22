package ml

import (
	"errors"
	"math/rand/v2"
	"os"
)

type Neuron struct {
	Weights []float64
	Bias    float64
}

type HiddenLayer struct {
	LayerNum uint
	Neurons  []Neuron
}

type OutputLayer struct {
	Neurons     []Neuron
	Logits      []float64
	ResultIndex int
}

// Hidden Layer constructor
func NewHiddenLayer(layerNum uint, neuronCount uint, paramsPerNeuron uint) (layer HiddenLayer, err error) {
	if neuronCount == 0 || paramsPerNeuron == 0 {
		err = errors.New("Arg neuronCount and paramsPerNeuron must at least be 1")
		return
	}

	layer = HiddenLayer{layerNum, make([]Neuron, neuronCount)}

	for i := range layer.Neurons {
		layer.Neurons[i].Weights = make([]float64, paramsPerNeuron)
	}
	return
}

// only to be used during training, populate hidden layer with random values for training
func (hl *HiddenLayer) InitWeights() {
	for i := range hl.Neurons {
		for weight := range hl.Neurons[i].Weights {
			hl.Neurons[i].Weights[weight] = rand.Float64()
		}
		hl.Neurons[i].Bias = rand.Float64()
	}
}

func (hl *HiddenLayer) LoadWeights(weightsFile *os.File) {
	// check if weights file exist
	// check if weights file is not empty
	// check if weights for this layer match the hyperparams in NewHiddenLayer
}
