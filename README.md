# goml
Neural Networks from scratch in Go (standard library only)

<b>Limitations: </b>
- No external libraries or modules
- For simplicity, I limit the input shape to (1,x)

<b>Goal: </b>
- Multilayer Perceptron (done)
- Binary classification (done)
- Support for multiple outputs (Multiple classification) (halted)
- Visualization (in progress)
</body>

<b>Current status:</b>
- Define basic data types (Input, Neuron, Layer)
- Calculate neuron outputs [type: (1,x) array of weight*input+bias]
- Define gradient methods and neural structure (heavily inspired by Karpathy's video)
- Backpropagate and train for k steps
- Produce and validate output
- More activation functions (Sigmoid and ReLU)
- Binary crossentropy

<b>[Goml Visualizer](https://github.com/vtech6/goml-visualizer) (in progress): </b><br><br>
![goml-visualizer](https://github.com/vtech6/goml-visualizer/blob/main/testVisualizer.gif) <br><br>
The Goml Visualizer was written using [Wails](https://wails.io) with React in Typescript for the frontend. The gif above shows the <b>test mode</b>, which allows you to browse your test set and see which values were correctly predicted by the model.<br><br>
<b>References and inspiration:</b>
- [Sebastian Lauge's Neural Networks Video](https://www.youtube.com/watch?v=hfMk-kjRv4c)
- [Andrej Karpathy's Micrograd Breakdown](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Aurelien Geron: Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems (March 13, 2017, O'Reilly)

<b>How to run</b>
- Make sure you have a valid Go installation by running `go version`
- Clone the repo
- Run with `go run .`
- The network parameters can be tweaked inside the `lib/network/run.go` file. The dataset labels can be changed inside `lib/network/utils.go`. Neuron activation functions can be found under `lib/network/neuron.go` and the engine blocks can be found under `lib/network/model.go`.
