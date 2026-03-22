package ml

import (
	"errors"
	"slices"
	"testing"
)

func TestIncompatibleMatrix(t *testing.T) {
	// NOTE: test case for when the matrix dimensions aren't m x n and n x p
	matrixA := [][]float64{
		{1, 3, 5},
		{7, 9, 11},
	}
	matrixB := [][]float64{
		{10, 9},
		{8, 7},
	}

	_, err := MatMul(matrixA, matrixB)

	if !errors.Is(err, ErrInvalidDemensions) {
		t.Errorf("got %v want ErrInvalidDimensions", err)
	}
}

func Test2x2with2x2Matrix(t *testing.T) {
	matrixA := [][]float64{
		{-1, 4},
		{2, 3},
	}

	matrixB := [][]float64{
		{0, -2},
		{5, 1},
	}

	expected := [][]float64{
		{20, 6},
		{15, -1},
	}

	result, err := MatMul(matrixA, matrixB)

	for currentSlice := range len(expected) {
		if !slices.Equal(result[currentSlice], expected[currentSlice]) {
			t.Errorf("got %v want %v", result, expected)
			break
		}
	}

	if err != nil {
		t.Errorf("got %v want nil", err)
	}
}

func Test2x3With3x2Matrix(t *testing.T) {
	matrixA := [][]float64{
		{2, -1, 3},
		{0, 4, -2},
	}
	matrixB := [][]float64{
		{1, 5},
		{-3, 2},
		{4, -1},
	}
	expected := [][]float64{
		{17, 5},
		{-20, 10},
	}

	result, err := MatMul(matrixA, matrixB)

	for currentSlice := range len(expected) {
		if !slices.Equal(result[currentSlice], expected[currentSlice]) {
			t.Errorf("got %v want %v", result, expected)
			break
		}
	}

	if err != nil {
		t.Errorf("got %v want nil", err)
	}
}

func Test1x3With3x1Matrix(t *testing.T) {
	matrixA := [][]float64{
		{5, -2, 1},
	}
	matrixB := [][]float64{
		{3},
		{0},
		{-4},
	}
	expected := [][]float64{
		{11},
	}

	result, err := MatMul(matrixA, matrixB)

	for currentSlice := range len(expected) {
		if !slices.Equal(result[currentSlice], expected[currentSlice]) {
			t.Errorf("got %v want %v", result, expected)
			break
		}
	}

	if err != nil {
		t.Errorf("got %v want nil", err)
	}
}

func Test3x1With1x4Matrix(t *testing.T) {
	matrixA := [][]float64{
		{-2},
		{3},
		{1},
	}
	matrixB := [][]float64{
		{4, -1, 0, 2},
	}
	expected := [][]float64{
		{-8, 2, 0, -4},
		{12, -3, 0, 6},
		{4, -1, 0, 2},
	}

	result, err := MatMul(matrixA, matrixB)

	for currentSlice := range len(expected) {
		if !slices.Equal(result[currentSlice], expected[currentSlice]) {
			t.Errorf("got %v want %v", result, expected)
			break
		}
	}

	if err != nil {
		t.Errorf("got %v want nil", err)
	}
}

func Test4x2With2x3Matrix(t *testing.T) {
	matrixA := [][]float64{
		{1, -3},
		{2, 0},
		{-1, 4},
		{3, 2},
	}
	matrixB := [][]float64{
		{2, -1, 5},
		{0, 3, -2},
	}
	expected := [][]float64{
		{2, -10, 11},
		{4, -2, 10},
		{-2, 13, -13},
		{6, 3, 11},
	}

	result, err := MatMul(matrixA, matrixB)

	for currentSlice := range len(expected) {
		if !slices.Equal(result[currentSlice], expected[currentSlice]) {
			t.Errorf("got %v want %v", result, expected)
			break
		}
	}

	if err != nil {
		t.Errorf("got %v want nil", err)
	}
}
