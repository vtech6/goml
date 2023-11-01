package network

import (
	"math/rand"
)

func Run() {
	//We seed the random generator seed to 42 to be able to reproduce results
	rand.Seed(42)

	//Tweak the params to see how they alter the accuracy of the network
	//Currently the network is training and predicting on the whole IRIS set.
	//Upcoming commits will handle train/test splitting of data and validation
	//with proper metric such as accuracy.
	trainTestSplitRatio := 0.8

	trainX, trainY, testX, testY := loadIrisData(trainTestSplitRatio)
	shape := []int{4, 4, 4, 1}
	learningRate := 0.001
	steps := 5
	batchSize := 15
	nEpochs := 5

	runNetwork(NetworkParams{
		nEpochs:      nEpochs,
		batchSize:    batchSize,
		trainX:       trainX,
		trainY:       trainY,
		testX:        testX,
		testY:        testY,
		shape:        shape,
		learningRate: learningRate,
		steps:        steps,
		verbose:      true,
	})
}
