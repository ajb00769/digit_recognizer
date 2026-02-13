package train

import (
	"allenbercero/digit_recognizer/ml"
	"math/rand"
)

// mutate hiddenLayer with random values to initalize for training
func InitHiddenLayerNeurons(hiddenLayer *ml.HiddenLayer) {
	for i := range hiddenLayer.Neurons {
		for weight := range hiddenLayer.Neurons[i].Weights {
			hiddenLayer.Neurons[i].Weights[weight] = rand.Float64()
		}
	}
	hiddenLayer.Bias = rand.Float64()
}
