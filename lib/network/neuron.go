package network

type Neuron struct {
	weights            []*Value
	bias               *Value
	activation         *Value
	activationFunction string
}

//We initialize each neuron with random weights in shape of the input and a
//random bias.

func (n *Neuron) initNeuron(inputLength int, neuronActivation string) {
	var value Value
	n.weights = make([]*Value, inputLength)
	for i := 0; i < inputLength; i++ {
		n.weights[i] = value.init(randomFloatStd())
	}
	n.bias = value.init(randomFloatStd())
	n.activationFunction = neuronActivation
}

//Multiply each element of the input data with its corresponding weight in the
//neuron. Add bias. Apply activation function.

func (n *Neuron) calculateOutput(neuronInput []float64) {
	var value Value
	activations := make([]*Value, len(n.weights))
	for i := 0; i < len(neuronInput); i++ {
		output := n.weights[i].multiply(value.init(neuronInput[i]))
		output = output.add(n.bias)
		activations[i] = output

	}
	activation := activations[0]
	for i := 1; i < len(activations); i++ {
		activation = activation.add(activations[i])
	}
	switch n.activationFunction {
	case "tanh":
		activation = activation.tanh()
	default:
		activation = activation.sigmoid()
	}
	n.activation = activation
}

//Deep neurons have an input of type Value, because they are fed the output of
//the previous layer of neurons.

func (n *Neuron) calculateOutputDeep(neuronInput []*Value) {
	activations := make([]*Value, len(n.weights))

	for i := 0; i < len(neuronInput); i++ {
		output := n.weights[i].multiply(neuronInput[i])
		output = output.add(n.bias)
		activations[i] = output
	}

	activation := activations[0]
	for i := 1; i < len(activations); i++ {
		if activations[i] != nil {
			activation = activation.add(activations[i])
		}
	}

	switch n.activationFunction {
	case "tanh":
		activation = activation.tanh()
	default:
		activation = activation.sigmoid()
	}
	n.activation = activation
}

func (n *Neuron) parameters() []*Value {
	params := make([]*Value, len(n.weights)+1)
	for i := 0; i < len(n.weights); i++ {
		params[i] = n.weights[i]
	}
	params[len(n.weights)] = n.bias
	return params
}

//The function below builds the tree of variables belonging to this output.

func buildTopo(v *Value, topo *[]*Value) {
	for i := 0; i < len(v.children); i++ {
		*topo = append(*topo, v.children[i])
		buildTopo(v.children[i], topo)
	}
}
