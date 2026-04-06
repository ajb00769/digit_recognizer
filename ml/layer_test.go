package ml

import (
	"math"
	"testing"
)

func TestNewHiddenLayerDimensions(t *testing.T) {
	layer, err := NewHiddenLayer(4, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(layer.Neurons) != 4 {
		t.Errorf("got %v neuron rows, want 4", len(layer.Neurons))
	}
	for i, neuron := range layer.Neurons {
		if len(neuron) != 3 {
			t.Errorf("neuron[%v] has %v weights, want 3", i, len(neuron))
		}
	}
	if len(layer.Bias) != 3 {
		t.Errorf("got %v biases, want 3", len(layer.Bias))
	}
}

func TestNewHiddenLayerZeroArgs(t *testing.T) {
	_, err := NewHiddenLayer(0, 5)
	if err == nil {
		t.Error("want error for neuronCount=0, got nil")
	}

	_, err = NewHiddenLayer(5, 0)
	if err == nil {
		t.Error("want error for weightsPerNeuron=0, got nil")
	}
}

func TestLayerInitWeightsInXavierRange(t *testing.T) {
	layer, _ := NewHiddenLayer(10, 10)
	layer.Init()

	xavierLimit := math.Sqrt(6.0 / (784.0 + 10.0))

	for i, neuron := range layer.Neurons {
		for j, w := range neuron {
			if w < -xavierLimit || w > xavierLimit {
				t.Errorf("weight[%v][%v] = %v out of xavier range [%v, %v]", i, j, w, -xavierLimit, xavierLimit)
			}
		}
	}
	for i, b := range layer.Bias {
		if b < -xavierLimit || b > xavierLimit {
			t.Errorf("bias[%v] = %v out of xavier range", i, b)
		}
	}
}

func TestLayerInitNotAllZeroes(t *testing.T) {
	layer, _ := NewHiddenLayer(5, 5)
	layer.Init()

	allZero := true
	for _, neuron := range layer.Neurons {
		for _, w := range neuron {
			if w != 0 {
				allZero = false
				break
			}
		}
	}
	if allZero {
		t.Error("all weights are zero after Init(), xavier init likely broken")
	}
}

func TestLayerForwardOutputLength(t *testing.T) {
	// 3 neurons, 4 weights each → forward with input of len 3 → output of len 4
	layer, _ := NewHiddenLayer(3, 4)
	layer.Init()

	input := []float64{0.1, 0.5, 0.9}
	result, err := layer.Forward(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 4 {
		t.Errorf("got output length %v, want 4", len(result))
	}
}

func TestLayerForwardOutputInSigmoidRange(t *testing.T) {
	layer, _ := NewHiddenLayer(3, 4)
	layer.Init()

	input := []float64{0.1, 0.5, 0.9}
	result, _ := layer.Forward(input)

	for i, v := range result {
		if v < 0 || v > 1 {
			t.Errorf("activated[%v] = %v, want value in [0, 1]", i, v)
		}
	}
}

func TestLayerForwardIncompatibleInput(t *testing.T) {
	// layer has 3 neuron rows, expects input of length 3, passing 5 should fail
	layer, _ := NewHiddenLayer(3, 4)
	layer.Init()

	badInput := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	_, err := layer.Forward(badInput)
	if err == nil {
		t.Error("want error for incompatible input length, got nil")
	}
}

func TestNewOutputLayerDimensions(t *testing.T) {
	prevLayer, _ := NewHiddenLayer(16, 8)
	output, err := NewOutputLayer(&prevLayer)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(output.CurrentLayer.Neurons) != 8 {
		t.Errorf("got %v output neuron rows, want 8", len(output.CurrentLayer.Neurons))
	}
	for i, neuron := range output.CurrentLayer.Neurons {
		if len(neuron) != 10 {
			t.Errorf("output neuron[%v] has %v weights, want 10", i, len(neuron))
		}
	}
	if len(output.CurrentLayer.Bias) != 10 {
		t.Errorf("got %v output biases, want 10", len(output.CurrentLayer.Bias))
	}
}

func TestOutputForwardSoftmaxSumsToOne(t *testing.T) {
	prevLayer, _ := NewHiddenLayer(16, 8)
	output, _ := NewOutputLayer(&prevLayer)
	output.Init()

	input := make([]float64, 8)
	for i := range input {
		input[i] = 0.5
	}

	result, err := output.Forward(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var sum float64
	for _, v := range result {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("softmax output sums to %v, want 1.0", sum)
	}
}

func TestOutputForwardReturns10Probs(t *testing.T) {
	prevLayer, _ := NewHiddenLayer(16, 8)
	output, _ := NewOutputLayer(&prevLayer)
	output.Init()

	input := make([]float64, 8)
	result, _ := output.Forward(input)

	if len(result) != 10 {
		t.Errorf("got %v output probabilities, want 10", len(result))
	}
}
