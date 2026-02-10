package ml

import (
	"math"
	"math/rand"
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

	result := Input(&testMatrix)

	for i := range expected {
		expected[i] = float64(i) / 100.0
	}

	if result != expected {
		t.Errorf("got %v, want %v", result, expected)
	}
}

func TestInputAllZeroes(t *testing.T) {
	result := Input(&emptyMatrix)
	if result != emptyArray {
		t.Errorf("got %v, want %v", result, emptyArray)
	}
}

func TestSoftmax(t *testing.T) {
	testCases := [][]float64{
		make([]float64, 3),
		make([]float64, 10),
		make([]float64, 100),
		make([]float64, 1000),
	}
	// NOTE: only testing up to 1000 neurons since this simple ML project's
	// output layer won't be dealing with a lot of neurons before output.

	// The number of neurons towards the end of the feed-forward would have
	// thinned out

	// Populate test data
	for testCase := range testCases {
		for i := range testCase {
			testCases[testCase][i] = rand.Float64()
		}
	}

	// Perform Tests
	for testCase := range testCases {
		var sum float64
		result := Softmax(testCases[testCase])

		for i := range result {
			sum += result[i]
		}

		// allow floating point inaccuracy up to 1e-10 epsilon value or 10 decimal places
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("got %v, want %v, in test case logit length %v", sum, 1, len(testCases[testCase]))
		}
	}
}

func TestSoftmaxLargeLogits(t *testing.T) {
	testCase := []float64{700.0, 0.22, 1.11}

	result := Softmax(testCase)

	for i := range result {
		if math.IsNaN(result[i]) {
			t.Errorf("got NaN, want a float64")
		}
	}
}

func TestSoftmaxNegativeLogits(t *testing.T) {
	var sum float64
	testCase := make([]float64, 5)

	// Populate test case with negative floats
	for i := range testCase {
		testCase[i] = rand.Float64() * -1
	}

	result := Softmax(testCase)

	for i := range result {
		sum += result[i]
		// check if value negative or exceeds 1 after softmax function
		if result[i] < 0 || result[i] > 1 {
			t.Errorf("got %v at index %v, want value between 0-1", result[i], i)
		}
	}

	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("got %v, want %v", sum, 1)
	}
}

func TestSoftmaxIdenticalLogits(t *testing.T) {
	testCase := []float64{5, 5, 5, 5}

	result := Softmax(testCase)

	for i := range result {
		if result[0] != result[i] {
			t.Errorf("got %v, want equal probabilities", result)
			break
		}
	}
}

func TestSoftmaxSingleLogit(t *testing.T) {
	testCase := []float64{0.333}

	result := Softmax(testCase)

	if result[0] != 1 {
		t.Errorf("want 100pct probability in single logit, got %v", result)
	}
}
