package train

import (
	"allenbercero/digit_recognizer/ml"
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"io"
	"log"
	"os"
)

func Train(trainingFile string, testingFile string, layerCount uint, neuronCount uint, weightCount uint) {
	// load filepath into File object
	trainingFileFile, err := os.Open(trainingFile)
	if err != nil {
		log.Fatalf("Failed to load training file. %v\n", err)
	}
	defer trainingFileFile.Close()

	testingFileFile, err := os.Open(testingFile)
	if err != nil {
		log.Fatalf("Failed to load testing file. %v\n", err)
	}
	defer testingFileFile.Close()

	trainingFileHash, err := getFileHash(trainingFileFile)
	if err != nil {
		log.Fatalf("Unable to get training file hash. %v\n", err)
	}
	trainingFileFile.Seek(0, io.SeekStart) // move file cursor back to beginning after hashing

	testingFileHash, err := getFileHash(testingFileFile)
	if err != nil {
		log.Fatalf("Unable to get testing file hash. %v\n", err)
	}
	testingFileFile.Seek(0, io.SeekStart) // move file cursor back to beginning after hashing

	// check if training and testing files have the same contents
	if bytes.Equal(trainingFileHash, testingFileHash) {
		log.Fatal("Training and testing files are the same (SHA-256 checksums match")
	}

	// load File into CSV reader
	trainingFileReader := csv.NewReader(trainingFileFile)
	testingFileReader := csv.NewReader(testingFileFile)

	// load file into workable string arrays / matrix ([][]string)
	trainingData, err := trainingFileReader.ReadAll()
	testingData, err := testingFileReader.ReadAll()

	// create hidden layers based on hyperparameters
	model, err := createHiddenLayers(layerCount, neuronCount, weightCount)
	if err != nil {
		log.Fatalf("Failed to create hidden layer. %s", err)
	}

	// initialize model with random weights and biases
	for i := range model {
		model[i].InitWeights()
	}

	// TODO: create output layer once forward propagation is reimplemented

	currentEpoch := 0
	currentEpochAccuracy := 0.0
	previousEpochAccuracy := 0.0

	maxEpochs := 1 // TODO: set this as hard limit just for testing, ideally we create a model accuracy check function

	for previousEpochAccuracy <= currentEpochAccuracy && currentEpoch < maxEpochs {
		previousEpochAccuracy = currentEpochAccuracy

		log.Printf("Epoch %d starting...\n", currentEpoch)
		trainModel(trainingData, &model)
		currentEpochAccuracy = testModel(testingData, &model)
		log.Printf("Epoch %d complete. Accuracy: %.4f\n", currentEpoch, currentEpochAccuracy)
		currentEpoch += 1
	}

	// - automatically train in epochs
	// - utilize computeLoss function to stop iterating new epochs
	//		when the current epoch's loss is greater than the previous epoch's
}

func trainModel(data [][]string, model *[]*ml.HiddenLayer) {
	// TODO: reimplement once forward propagation is rewritten
}

// returns model accuracy relative to testing data
func testModel(data [][]string, model *[]*ml.HiddenLayer) float64 {
	// forward pass and get predictions
	// compare against labels and count how many are correct
	// return accuracy percentage
	return 0.0
}

// stream file and hash per chunk, use previous chunk as input for the current chunk, rinse and repeat
func getFileHash(file *os.File) ([]byte, error) {
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return nil, err
	}
	return hash.Sum(nil), nil
}

func createHiddenLayers(layerCount uint, neuronCount uint, weightCount uint) (hiddenLayers []*ml.HiddenLayer, err error) {
	for layerNum := range layerCount {
		paramsPerNeuron := weightCount
		if layerNum > 0 {
			paramsPerNeuron = neuronCount // subsequent layers receive previous layer's neuron outputs
		}
		layer, err := ml.NewHiddenLayer(layerNum, neuronCount, paramsPerNeuron)

		if err != nil {
			return nil, err
		}

		hiddenLayers = append(hiddenLayers, &layer)
	}
	return
}

// TODO: reimplement once output layer and forward propagation are rewritten

func loadTrainingData(file []byte, model []*ml.HiddenLayer) {
}

func loadTestingData(file []byte) {
}

func computeLoss() float32 { return 0.0 }
func backPropagate()       {}
func testTrainedModel()    {} // will include benchmarking created model/weights
func saveEpoch()           {}

// create dataframe struct that will house the training and testing data
// load training and testing data from url into memory via class method for dataframe
// check if data in memory is of type csv
// initialize initial model weights with random values
// create function that trains 1 epoch
// store result of previous epoch
// compute loss from previous epoch on 2nd and onward iteration
// create and run backpropagation function to adjust weights
// stop epoch training loop when loss increases vs previous iteration
// save final model weights into file
