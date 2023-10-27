package network

func Run() {
	inputs, targets := loadIrisData()
	shape := []int{4, 3, 3, 1}
	learningRate := 0.001
	steps := 5

	runNetwork(NetworkParams{
		trainX:       inputs,
		trainY:       targets,
		shape:        shape,
		learningRate: learningRate,
		steps:        steps,
	})

}
