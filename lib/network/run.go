package network

import "fmt"

func Run() {
	feedForward()
}

func feedForward() {
	inputs := [][]float64{{2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}}
	targets := []float64{1.0, -1.0, -1.0, 1.0}
	network := MLP{}
	network.initNetwork([]int{3, 4, 4, 1})
	//For step in steps
	for step := 0; step < 20; step++ {
		var value Value
		loss := value.init(0.0)
		outputs := make([]*Value, len(inputs))
		//For input of inputs
		for inputIndex := 0; inputIndex < len(inputs); inputIndex++ {
			//Calculate output and add its value to outputs
			output := network.calculateOutput(inputs[inputIndex])
			outputs[inputIndex] = output[0]
		}

		for outputIndex := 0; outputIndex < len(outputs); outputIndex++ {
			targetValue := value.init(targets[outputIndex])
			negativeOutput := outputs[outputIndex].negative()
			yDifference := negativeOutput.add(targetValue)
			loss = loss.add(yDifference.tanh())

		}

		loss.gradient = 1.0
		loss.backward()
		loss.calculateGradients()
		parameters := network.parameters()
		for layerIndex := 0; layerIndex < len(parameters); layerIndex++ {
			for valueIndex := 0; valueIndex < len(parameters[layerIndex]); valueIndex++ {
				for valueIndex2 := 0; valueIndex2 < len(parameters[layerIndex][valueIndex]); valueIndex2++ {
					value := parameters[layerIndex][valueIndex][valueIndex2]
					value.value += loss.value * value.gradient
				}
			}
		}
		fmt.Println("---------")
		fmt.Println("Loss: ", loss.value)
		for i := 0; i < len(outputs); i++ {
			fmt.Println("Prediction", i, " ", outputs[i].value)
		}
		fmt.Println(parameters[0][0][3].gradient, "PARAM")

	}
}
func workingNet() {

	network := MLP{}
	network.initNetwork([]int{3, 4, 4, 1})
	output := network.calculateOutput([]float64{1.0, -0.5, 2.0})
	var value Value
	targetY := value.init(1.0)
	negativeOutput := output[0].negative()
	loss := negativeOutput.add(targetY)
	loss = loss.tanh()
	loss.gradient = 1.0
	loss.backward()
	negativeOutput.backward()
	loss.calculateGradients()
	fmt.Println(loss.gradient, loss.children[0].gradient, network.layers[1].neurons[0].activation.gradient, network.layers[0].neurons[0].activation.gradient)
}
