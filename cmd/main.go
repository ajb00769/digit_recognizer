package main

import (
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

	result := matMul(matrixA, matrixB)

	fmt.Println()
	fmt.Printf("%v\n", result)
}

func createMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)

	for rowNum := range len(matrix) {
		matrix[rowNum] = make([]float64, cols)
	}

	return matrix
}

func matMul(matrixA, matrixB [][]float64) [][]float64 {
	result := createMatrix(len(matrixA), len(matrixB[0]))

	for i := range matrixA {
		currentMatrixARow := matrixA[i]

		fmt.Printf("%v", currentMatrixARow)
		fmt.Println()

		currentMatrixBColumnValues := make([]float64, len(matrixB))

		for j := range matrixB {
			currentMatrixBColumnValues[j] = matrixB[j][i]
		}

		fmt.Printf("%v", currentMatrixBColumnValues)
		fmt.Println()
	}

	return result
}

func arraySum(arr []float64) float64 {
	var sum float64 = 0.0

	for i := range arr {
		sum += arr[i]
	}

	return sum
}
