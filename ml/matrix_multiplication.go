package ml

import "log"

func MatMul(matrixA, matrixB [][]float64) [][]float64 {
	if len(matrixA[0]) != len(matrixB) {
		// TODO: replace error handling without crashing app
		log.Fatal("[ERROR] Matrix A's columns must be equal to Matrix B's rows.\n")
	}

	result := createMatrix(len(matrixA), len(matrixB[0]))

	for i := range matrixA {
		currentMatrixARow := matrixA[i]
		currentMatrixBColumnValues := make([]float64, len(matrixB))

		for j := range len(matrixB[0]) {
			for k := range matrixB {
				currentMatrixBColumnValues[k] = matrixB[k][j]
			}

			sumOfArrayProducts := arraySum(multArrays(currentMatrixARow, currentMatrixBColumnValues))
			result[i][j] = sumOfArrayProducts
		}
	}

	return result
}

func createMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)

	for rowNum := range len(matrix) {
		matrix[rowNum] = make([]float64, cols)
	}

	return matrix
}

func multArrays(arr1, arr2 []float64) []float64 {
	result := make([]float64, len(arr1))

	for i := range len(arr1) {
		result[i] = arr1[i] * arr2[i]
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
