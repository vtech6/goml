package network

type Neuron struct {
	weights    []*Value
	bias       *Value
	output     []float64
	delta      float64
	activation *Value
}

func (n *Neuron) initNeuron(inputLength int) {
	var value Value
	n.weights = make([]*Value, inputLength)
	for i := 0; i < inputLength; i++ {
		n.weights[i] = value.init(randomFloatStd())
	}
	n.bias = value.init(randomFloatStd())
	n.output = make([]float64, inputLength)
}

func (n *Neuron) calculateOutput(neuronInput []float64) {
	var value Value
	var activation Value
	for i := 0; i < len(neuronInput); i++ {
		output := n.weights[i].multiply(value.init(neuronInput[i]))
		output = output.add(n.bias)
		activation = *output.tanh()

	}
	n.activation = &activation
}

func (n *Neuron) calculateOutputDeep(neuronInput []*Value) {
	var activation Value
	for i := 0; i < len(neuronInput); i++ {

		output := n.weights[i].multiply(neuronInput[i])
		output = output.add(n.bias)
		activation = *output.tanh()
	}
	n.activation = &activation
}

func (n *Neuron) parameters() *[]*Value {
	params := make([]*Value, len(n.weights)+1)
	for i := 0; i < len(n.weights); i++ {
		params[i] = n.weights[i]
	}
	params[len(n.weights)] = n.bias
	return &params
}

func buildTopo(v *Value, topo *[]*Value) {
	_topo := *topo
	presentInTopo := false
	for i := 0; i < len(_topo); i++ {
		for j := 0; j < len(v.children); j++ {
			if _topo[i] == v.children[j] {
				presentInTopo = true
			}
		}
	}
	for i := 0; i < len(v.children); i++ {
		if presentInTopo == false {
			*topo = append(*topo, v.children[i])
			buildTopo(v.children[i], topo)
		}
	}
}
