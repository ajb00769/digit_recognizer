package train

import "testing"

func TestInitNeuronHappyPath(t *testing.T) {
	neurons := 12
	parameters := 36

	result := InitNeuron(neurons, parameters)

	if len(result) != neurons {
		t.Errorf("got %v neurons, want %v", len(result), neurons)
	}

	for i := range result {
		if len(result[i].Weights) != parameters {
			t.Errorf("neuron %v: got %v weights, want %v", i, len(result[i].Weights), parameters)
		}
	}
}

// Test if initialized neuron's weight is between 0-1
func TestInitNeuronWeightsInRange(t *testing.T) {
	result := InitNeuron(10, 20)

	for i, n := range result {
		for j, w := range n.Weights {
			if w < 0 || w >= 1 {
				t.Errorf("neuron %v weight %v: got %v, want value in [0, 1)", i, j, w)
			}
		}
	}
}

// Test initialized random bias is between 0-1
func TestInitNeuronBiasInRange(t *testing.T) {
	result := InitNeuron(10, 20)

	for i, n := range result {
		if n.Bias < 0 || n.Bias >= 1 {
			t.Errorf("neuron %v: got bias %v, want value in [0, 1)", i, n.Bias)
		}
	}
}

// Edge case if only 1 neuron and 784 weights is chosen as hyperparameters
func TestInitNeuronSingleNeuron(t *testing.T) {
	result := InitNeuron(1, 784)

	if len(result) != 1 {
		t.Errorf("got %v neurons, want 1", len(result))
	}

	if len(result[0].Weights) != 784 {
		t.Errorf("got %v weights, want 784", len(result[0].Weights))
	}
}

// Edge case if only 1 weight per neuron is chosen as hyperparameters
func TestInitNeuronSingleParam(t *testing.T) {
	result := InitNeuron(5, 1)

	for i, n := range result {
		if len(n.Weights) != 1 {
			t.Errorf("neuron %v: got %v weights, want 1", i, len(n.Weights))
		}
	}
}

// Boundary case where 0 neurons is decided by designer as hyperparameters
func TestInitNeuronZeroNeurons(t *testing.T) {
	result := InitNeuron(0, 10)

	if len(result) != 0 {
		t.Errorf("got %v neurons, want 0", len(result))
	}
}
