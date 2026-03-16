package main

import (
	"allenbercero/digit_recognizer/train"
	"fmt"
)

func main() {
	trainingFile := "files/mnist_train.csv"
	testingFile := "files/mnist_test.csv"

	layerCount := uint(2)
	neuronCount := uint(16)
	weightCount := uint(784)

	fmt.Println("Starting training...")
	train.Train(trainingFile, testingFile, layerCount, neuronCount, weightCount)
	fmt.Println("Training complete.")
}
