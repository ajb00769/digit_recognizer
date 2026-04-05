package ml

import (
	"encoding/json"
	"fmt"
)

type NetworkConfig struct {
	HiddenLayerSizes []int // hidden layer sizes only
}

type NeuralNetwork struct {
	HiddenLayers []Layer
	OutputLayer  Output
}

func NewNetwork(conf NetworkConfig) (network NeuralNetwork, err error) {
	layers := len(conf.HiddenLayerSizes)

	network.HiddenLayers = make([]Layer, layers)

	for layer := range layers {
		if layer == 0 {
			network.HiddenLayers[layer], err = NewHiddenLayer(784, conf.HiddenLayerSizes[0])
			continue
		}
		network.HiddenLayers[layer], err = NewHiddenLayer(conf.HiddenLayerSizes[layer-1], conf.HiddenLayerSizes[layer])
	}

	network.OutputLayer, err = NewOutputLayer(&network.HiddenLayers[layers-1])
	return
}

// Weight and bias initialization function for training
func (nn *NeuralNetwork) Init() {
	for layer := range nn.HiddenLayers {
		nn.HiddenLayers[layer].Init()
	}

	nn.OutputLayer.Init()
}

// TODO
func (nn *NeuralNetwork) Load(fp string) {}

func (nn *NeuralNetwork) ForwardPropagate(inp Input) (err error) {
	signal := inp.Flatten()

	for layer := range nn.HiddenLayers {
		signal, err = nn.HiddenLayers[layer].Forward(signal)
		data, _ := json.MarshalIndent(signal, "", "  ")
		fmt.Println(string(data))
	}

	if err != nil {
		return err
	}

	softmaxed, err := nn.OutputLayer.Forward(signal)

	data, _ := json.MarshalIndent(softmaxed, "", "  ")
	fmt.Println(string(data))

	return nil
}

func (nn *NeuralNetwork) BackPropagate() {}
