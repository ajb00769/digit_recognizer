package ml

import "math"

// TODO: Input Layer 2D array of 28 * 28
func input([784]float64) []float64 {
	// Implement input layer logic here
	var output []float64
	return output
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
	return 1 / (1 + math.Pow(math.E, -x))
}

// TODO: Matrix multiplication
func matMul(matrixA, matrixB [][]float64) [][]float64 {
	var output [][]float64

	/*
		for rowA := 0; rowA < len(matrixA); rowA++ {
		for columnA := 0; columnA < len(matrixA[0])
		}*/

	return output
}
