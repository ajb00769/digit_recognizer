package ml

import (
	"os"
	"slices"
	"testing"
)

var testMatrix [28][28]float64
var emptyMatrix [28][28]float64

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
	inp := Input{Raw: testMatrix}
	expected := make([]float64, 784)

	for i := range expected {
		expected[i] = float64(i) / 100.0
	}

	result := inp.Flatten()

	if !slices.Equal(result, expected) {
		t.Errorf("got %v, want %v", result, expected)
	}
}

func TestInputAllZeroes(t *testing.T) {
	inp := Input{Raw: emptyMatrix}
	result := inp.Flatten()
	expected := make([]float64, 784)

	if !slices.Equal(result, expected) {
		t.Errorf("got %v, want %v", result, expected)
	}
}
