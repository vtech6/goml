package network

func Run() {

	//Tweak the params to see how they alter the accuracy of the network
	//Currently the network is training and predicting on the whole IRIS set.
	//Upcoming commits will handle train/test splitting of data and validation
	//with proper metric such as accuracy.

	inputs, targets := loadIrisData()
	shape := []int{4, 3, 3, 1}
	learningRate := 0.001
	steps := 5
	batchSize := 15
	nEpochs := 5

	runNetwork(NetworkParams{
		nEpochs:      nEpochs,
		batchSize:    batchSize,
		trainX:       inputs,
		trainY:       targets,
		shape:        shape,
		learningRate: learningRate,
		steps:        steps,
	})

}
