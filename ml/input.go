package ml

// Input Layer 28 * 28 matrix flatten. We do not mutate the Input so
// it's safe to pass in the Input parameter value itself instead of
// creating a copy of it
func FlattenInput(matrixInput *[28][28]float64) [784]float64 {
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
