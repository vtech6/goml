package network

func runEpochs(epochs int, function func(int)) {
	for i := 0; i < epochs; i++ {
		function(i)
	}
}

func getMiniBatches(batchSize int, inputs [][]float64, targets [][]float64) ([][][]float64, [][][]float64) {
	nBatches := len(inputs) / batchSize
	_inputs := make([][][]float64, 0)
	_targets := make([][][]float64, 0)
	for i := 0; i < nBatches; i++ {
		firstIndex := i * batchSize
		lastIndex := (i + 1) * batchSize
		_inputs = append(_inputs, inputs[firstIndex:lastIndex])
		_targets = append(_targets, targets[firstIndex:lastIndex])
	}
	return _inputs, _targets
}
