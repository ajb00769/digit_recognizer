package ml

import (
	"math"
	"slices"
)

// Activation Functions
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Softmax(logits []float64) []float64 {
	// get max value in slice for exponent rule
	max := slices.Max(logits)

	var total float64 = 0.0
	output := make([]float64, len(logits))

	for i := range len(logits) {
		// populate the output slice with all numerators to be divided by the total
		output[i] = math.Exp(logits[i] - max)
		total += output[i]
	}

	// perform softmax exponentiated numerator divided by total and replace
	// the numerator with the final value
	for i := range len(output) {
		output[i] /= total
	}

	return output
}
