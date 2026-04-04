package ml

type NetworkConfig struct {
	InputSize        int
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
			network.HiddenLayers[layer], err = NewHiddenLayer(conf.HiddenLayerSizes[0], 10)
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
func (nn *NeuralNetwork) LoadModel(fp string) {}
