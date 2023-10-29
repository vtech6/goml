package network

import (
	"fmt"
	"math/rand"
)

func Run() {
	//We seed the random generator seed to 42 to be able to reproduce results
	rand.Seed(42)

	//Tweak the params to see how they alter the accuracy of the network
	//Currently the network is training and predicting on the whole IRIS set.
	//Upcoming commits will handle train/test splitting of data and validation
	//with proper metric such as accuracy.

	trainX, trainY, testX, testY := loadIrisData()
	shape := []int{4, 4, 4, 3}
	learningRate := 0.001
	steps := 5
	batchSize := 15
	nEpochs := 2

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
	array1 := array()
	array2 := make([][]*Value, 1)
	array2[0] = array1
	fmt.Println(array2[0][0].value)
}

func array() []*Value {
	var value *Value
	value1 := value.init(1)
	value2 := value.init(2)
	return []*Value{value1, value2}
}
