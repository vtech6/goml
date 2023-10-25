package network

func Run() {

	inputs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0}}

	targets := []float64{1.0, -1.0, -1.0, 1.0}
	shape := []int{3, 4, 4, 1}
	learningRate := 0.01
	steps := 100

	runNetwork(NetworkParams{
		trainX:       inputs,
		trainY:       targets,
		shape:        shape,
		learningRate: learningRate,
		steps:        steps,
	})
}
