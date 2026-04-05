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
		network.HiddenLayers[layer], err = NewHiddenLayer(conf.HiddenLayerSizes[layer], conf.HiddenLayerSizes[layer-1])
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

// WIP
func (nn *NeuralNetwork) ForwardPropagate(inp Input) (err error) {
	inp.Flatten()

	result, err := MatMul(inp.Flattened, nn.HiddenLayers[0].Neurons)
	data, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println("MATMUL RESULT INPUT -> FIRST HIDDEN LAYER:")
	fmt.Println(string(data))

	inputResult := make([]float64, len(result[0]))
	for num := range result[0] {
		inputResult[num] = result[0][num] + nn.HiddenLayers[0].Bias[num]
	}
	data, _ = json.MarshalIndent(inputResult, "", "  ")
	fmt.Println("FIRST HIDDEN LAYER WEIGHTS + BIAS:")
	fmt.Println(string(data))

	activated := make([]float64, len(inputResult))
	for item := range inputResult {
		activated[item] = Sigmoid(inputResult[item])
	}
	data, _ = json.MarshalIndent(activated, "", "  ")
	fmt.Println("FIRST HIDDEN LAYER ACTIVATED:")
	fmt.Println(string(data))

	return nil
}
