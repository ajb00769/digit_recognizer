package train

import (
	"allenbercero/digit_recognizer/ml"
	"testing"
)

// helper to create a HiddenLayer with neurons and weights already allocated
func createTestLayer(t *testing.T, layerNum uint, neuronCount uint, paramsPerNeuron uint) ml.HiddenLayer {
	layer, err := ml.NewHiddenLayer(layerNum, neuronCount, paramsPerNeuron)
	if err != nil {
		t.Fatalf("failed to create test layer: %v", err)
	}
	return layer
}

func TestInitHiddenLayerNeuronsHappyPath(t *testing.T) {
	layer := createTestLayer(t, 0, 12, 36)

	InitHiddenLayerNeurons(&layer)

	if len(layer.Neurons) != 12 {
		t.Errorf("got %v neurons, want 12", len(layer.Neurons))
	}

	for i := range layer.Neurons {
		if len(layer.Neurons[i].Weights) != 36 {
			t.Errorf("neuron %v: got %v weights, want 36", i, len(layer.Neurons[i].Weights))
		}
	}
}

// Test if initialized neuron's weight is between 0-1
func TestInitHiddenLayerNeuronsWeightsInRange(t *testing.T) {
	layer := createTestLayer(t, 0, 10, 20)

	InitHiddenLayerNeurons(&layer)

	for i, n := range layer.Neurons {
		for j, w := range n.Weights {
			if w < 0 || w >= 1 {
				t.Errorf("neuron %v weight %v: got %v, want value in [0, 1)", i, j, w)
			}
		}
	}
}

// Test initialized random bias is between 0-1
func TestInitHiddenLayerNeuronsBiasInRange(t *testing.T) {
	layer := createTestLayer(t, 0, 10, 20)

	InitHiddenLayerNeurons(&layer)

	if layer.Bias < 0 || layer.Bias >= 1 {
		t.Errorf("got bias %v, want value in [0, 1)", layer.Bias)
	}
}

// Edge case if only 1 neuron and 784 weights is chosen as hyperparameters
func TestInitHiddenLayerNeuronsSingleNeuron(t *testing.T) {
	layer := createTestLayer(t, 0, 1, 784)

	InitHiddenLayerNeurons(&layer)

	if len(layer.Neurons) != 1 {
		t.Errorf("got %v neurons, want 1", len(layer.Neurons))
	}

	if len(layer.Neurons[0].Weights) != 784 {
		t.Errorf("got %v weights, want 784", len(layer.Neurons[0].Weights))
	}
}

// Edge case if only 1 weight per neuron is chosen as hyperparameters
func TestInitHiddenLayerNeuronsSingleParam(t *testing.T) {
	layer := createTestLayer(t, 0, 5, 1)

	InitHiddenLayerNeurons(&layer)

	for i, n := range layer.Neurons {
		if len(n.Weights) != 1 {
			t.Errorf("neuron %v: got %v weights, want 1", i, len(n.Weights))
		}
	}
}
