package main

import (
	"allenbercero/digit_recognizer/ml"
	"fmt"
)

func main() {
	matrixA := [][]float64{
		{2, -1, 3},
		{0, 4, -2},
	}

	matrixB := [][]float64{
		{1, 5},
		{-3, 2},
		{4, -1},
	}

	result, err := ml.MatMul(matrixA, matrixB)

	if err == nil {
		fmt.Printf("%v\n", result)
	}
}
