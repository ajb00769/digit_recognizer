package ml

import (
	"math"
)

// Input Layer 28 * 28 matrix flatten
func input(matrixInput *[28][28]float64) [784]float64 {
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

// TODO: Hidden Layer
func layerOne() {
	// Implement hidden layer logic here
}

func layerTwo() {
	// Implement hidden layer logic here
}

// TODO: Output Layer
func output() {
	// Implement output layer logic here
}

// TODO: Activation Functions
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
