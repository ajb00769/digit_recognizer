package ml

// Input Layer 28 * 28 matrix flatten. We do not mutate the Input so
// it's safe to pass in the Input parameter value itself instead of
// creating a copy of it

type Input struct {
	Raw       [28][28]float64
	Flattened [784]float64
}

func (inp *Input) Flatten() {
	// NOTE: using fixed size array to enforce rigid input lengths
	// anything less or greater than the fixed size should not be
	// acceptable since the feed-forward mechanism to other layers
	// will need this fixed sizing
	for row := range inp.Raw {
		for item := range inp.Raw[row] {
			// hardcoded 28 in the formula since we expect 28 to be
			// a constant size
			inp.Flattened[(row*28)+item] = inp.Raw[row][item]
		}
	}
}
