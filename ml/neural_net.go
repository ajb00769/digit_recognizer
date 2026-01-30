package ml

import "math"

const EULER float64 = 2.718281828459045

// TODO: Input Layer 2D array of 28 * 28
func input([784]float32) []float32 {
	// Implement input layer logic here
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
	return 1 / (1 + (EULER * math.Exp(-x)))
}

// TODO: Matrix multiplication
func matMul(matrixA [][]float32, matrixB [][]float32) [][]float32 {
	// Implement
	return nil
}
