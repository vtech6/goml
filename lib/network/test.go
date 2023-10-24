package network

import "fmt"

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
	aoutput := output.tanh()
	aoutput.gradient = 1.0
	aoutput.backward()
	aoutput.calculateGradients()
	fmt.Println(aoutput.value)
	fmt.Println(aoutput.gradient, output.gradient, bias.gradient, abcd.gradient, d.gradient, c.gradient, b.gradient, a.gradient)

}

func neuronTest() {
	a := Neuron{}
	a.initNeuron(3)
	a.calculateOutput([]float64{1.0, -1.0, 2.0})
	output := a.activation.tanh()
	output.gradient = 1.0
	output.backward()
	output.calculateGradients()
	fmt.Println(output.gradient, output.children[0].gradient, output.children[0].children[1].gradient)
}

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
