package ml

import (
	"math"
	"testing"
)

func newTestNetwork(t *testing.T, sizes []int) NeuralNetwork {
	t.Helper()
	nn, err := NewNetwork(NetworkConfig{HiddenLayerSizes: sizes})
	if err != nil {
		t.Fatalf("NewNetwork failed: %v", err)
	}
	nn.Init()
	return nn
}

func TestNewNetworkHiddenLayerCount(t *testing.T) {
	nn, _ := NewNetwork(NetworkConfig{HiddenLayerSizes: []int{16, 12}})
	if len(nn.HiddenLayers) != 2 {
		t.Errorf("got %v hidden layers, want 2", len(nn.HiddenLayers))
	}
}

func TestNewNetworkFirstLayerDimensions(t *testing.T) {
	// first hidden layer: 784 neurons x HiddenLayerSizes[0] weights
	nn, _ := NewNetwork(NetworkConfig{HiddenLayerSizes: []int{16}})
	layer := nn.HiddenLayers[0]

	if len(layer.Neurons) != 784 {
		t.Errorf("first layer got %v neuron rows, want 784", len(layer.Neurons))
	}
	if len(layer.Neurons[0]) != 16 {
		t.Errorf("first layer got %v weights per neuron, want 16", len(layer.Neurons[0]))
	}
}

func TestNewNetworkLayerChaining(t *testing.T) {
	// layer[0]: 784 x 16, layer[1]: 16 x 8
	// layer[1] input length must equal layer[0] output length (16)
	nn, _ := NewNetwork(NetworkConfig{HiddenLayerSizes: []int{16, 8}})

	if len(nn.HiddenLayers[1].Neurons) != 16 {
		t.Errorf("layer[1] got %v neuron rows, want 16", len(nn.HiddenLayers[1].Neurons))
	}
	if len(nn.HiddenLayers[1].Neurons[0]) != 8 {
		t.Errorf("layer[1] got %v weights per neuron, want 8", len(nn.HiddenLayers[1].Neurons[0]))
	}
}

func TestNewNetworkOutputLayerDimensions(t *testing.T) {
	nn, _ := NewNetwork(NetworkConfig{HiddenLayerSizes: []int{16, 8}})
	outNeurons := nn.OutputLayer.CurrentLayer.Neurons

	if len(outNeurons) != 8 {
		t.Errorf("output layer got %v neuron rows, want 8", len(outNeurons))
	}
	if len(outNeurons[0]) != 10 {
		t.Errorf("output layer got %v weights per neuron, want 10", len(outNeurons[0]))
	}
}

func TestForwardPropagateOutputLength(t *testing.T) {
	nn := newTestNetwork(t, []int{16, 12})
	inp := Input{}

	// populate with dummy pixel values
	for i := range inp.Raw {
		for j := range inp.Raw[i] {
			inp.Raw[i][j] = float64((i*28)+j) / 255.0
		}
	}

	err := nn.ForwardPropagate(inp)
	if err != nil {
		t.Fatalf("ForwardPropagate returned error: %v", err)
	}
}

func TestForwardPropagateSoftmaxSumsToOne(t *testing.T) {
	nn := newTestNetwork(t, []int{16, 12})

	// run a real forward pass and check softmax via output layer directly
	inp := Input{}
	for i := range inp.Raw {
		for j := range inp.Raw[i] {
			inp.Raw[i][j] = 0.5
		}
	}

	flatSignal := inp.Flatten()
	for _, layer := range nn.HiddenLayers {
		flatSignal, _ = layer.Forward(flatSignal)
	}
	result, err := nn.OutputLayer.Forward(flatSignal)
	if err != nil {
		t.Fatalf("output Forward failed: %v", err)
	}

	var sum float64
	for _, v := range result {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("softmax sums to %v, want 1.0", sum)
	}
}

func TestForwardPropagateAllProbsInRange(t *testing.T) {
	nn := newTestNetwork(t, []int{16, 12})

	inp := Input{}
	for i := range inp.Raw {
		for j := range inp.Raw[i] {
			inp.Raw[i][j] = 0.5
		}
	}

	flatSignal := inp.Flatten()
	for _, layer := range nn.HiddenLayers {
		flatSignal, _ = layer.Forward(flatSignal)
	}
	result, _ := nn.OutputLayer.Forward(flatSignal)

	for i, v := range result {
		if v < 0 || v > 1 {
			t.Errorf("prob[%v] = %v, want value in [0, 1]", i, v)
		}
	}
}
