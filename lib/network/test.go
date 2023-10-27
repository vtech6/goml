package network

import "fmt"

//The following functions help me verify integrity of the code. Diagnosing bugs
//becomes less of a hassle when I can run subcomponents of the network separately.

//Calculate gradient after a chain of arithmetic operations.

func valueTest() {
	var value Value
	a := value.init(2)
	b := value.init(-3)
	c := value.init(1)
	d := value.init(0)
	bias := value.init(6.88137358701954)
	ab := a.multiply(b)
	cd := c.multiply(d)
	abcd := ab.add(cd)
	output := abcd.add(bias)
	activeOutput := output.tanh()
	activeOutput.gradient = 1.0
	activeOutput.backward()
	activeOutput.calculateGradients()
	fmt.Println(activeOutput.value)
	fmt.Println(activeOutput.gradient, output.gradient, bias.gradient, abcd.gradient, d.gradient, c.gradient, b.gradient, a.gradient)
}

//Calculate gradient after a chain of operations on neurons.

func neuronTest() {
	a := Neuron{}
	a.initNeuron(3)
	a.calculateOutput([]float64{1.0, -1.0, 2.0})

	b := Neuron{}
	b.initNeuron(1)
	b.calculateOutputDeep([]*Value{a.activation})

	output := b.activation
	output.gradient = 1.0
	output.backward()
	output.calculateGradients()
	for i := 0; i < 3; i++ {
		fmt.Println("AW", i, ":", a.weights[i].gradient)
	}

	fmt.Println("BW", 0, ":", b.weights[0].gradient)
	fmt.Println("BIAS:", a.bias)
	fmt.Println("Activation:", a.activation.gradient)

}

//Calculate gradient after a chain of layer operations.

func layerTest() {
	var _layer Layer
	layer := _layer.initLayer(3, 3)
	fmt.Println(layer)
	layer.feedForward([]float64{1.0, -1.0, 2.0})
	fmt.Println(layer.output)
	layer2 := _layer.initLayer(3, 1)
	layer2.feedForwardDeep(layer.output)
	var value Value
	loss := value.init(0.0)
	for i := 0; i < len(layer2.output); i++ {
		loss = loss.add(layer2.output[i])
	}
	loss = loss.tanh()
	loss.gradient = 1.0
	loss.backward()
	loss.calculateGradients()
	fmt.Println(loss.gradient, loss.children[0].gradient, layer2.neurons[0].activation.gradient, layer.neurons[0].activation.gradient)
}

func layerTest2() {
	inputs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0}}

	targets := []float64{1.0, -1.0, -1.0, 1.0}
	var _layer Layer
	layer := _layer.initLayer(3, 3)
	layer2 := _layer.initLayer(3, 1)
	var value Value
	for step := 0; step < 10; step++ {

		loss := value.init(0.0)
		for i := 0; i < len(inputs); i++ {
			layer.feedForward([]float64{1.0, -1.0, 2.0})
			layer2.feedForwardDeep(layer.output)

			output := layer2.output[0]
			diff := value.init(targets[i]).add(output.multiply(value.init(-1)))
			loss = diff.square().add(loss)
		}
		parameters := make([]*Value, 0)
		l1p := layer.parameters()
		for i := 0; i < len(l1p); i++ {
			parameters = append(parameters, l1p[i])
		}
		l2p := layer2.parameters()
		for i := 0; i < len(l2p); i++ {
			parameters = append(parameters, l2p[i])
		}

		for paramI := 0; paramI < len(parameters); paramI++ {
			fmt.Println(parameters[paramI].gradient)
			parameters[paramI].gradient = 0

		}
		loss.gradient = 1.0
		loss.backward()
		loss.calculateGradients()
		for paramI := 0; paramI < len(parameters); paramI++ {
			parameters[paramI].value += parameters[paramI].gradient
		}

		fmt.Println(step, loss.value)
	}

}

func mlpTest() {

	inputs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0}}

	targets := [][]float64{{1.0}, {-1.0}, {-1.0}, {1.0}}
	shape := []int{3, 4, 4, 1}
	learningRate := 0.01
	steps := 100
	batchSize := 1
	nEpochs := 2

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
