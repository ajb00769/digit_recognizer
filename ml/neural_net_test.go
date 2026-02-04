package ml

import (
	"os"
	"testing"
)

var testMatrix [28][28]float64
var emptyMatrix [28][28]float64
var emptyArray [784]float64

func populateTestMatrix() {
	for i := range testMatrix {
		for j := range testMatrix[i] {
			testMatrix[i][j] = ((float64(i) * 28.0) + float64(j)) / 100.0
		}
	}
}

func TestMain(m *testing.M) {
	populateTestMatrix()
	exitCode := m.Run()
	os.Exit(exitCode)
}

func TestInputHappyPath(t *testing.T) {
	var expected [784]float64

	result := input(&testMatrix)

	for i := range expected {
		expected[i] = float64(i) / 100.0
	}

	if result != expected {
		t.Errorf("got %v, want %v", result, expected)
	}
}

func TestInputAllZeroes(t *testing.T) {
	result := input(&emptyMatrix)
	if result != emptyArray {
		t.Errorf("got %v, want %v", result, emptyArray)
	}
}
