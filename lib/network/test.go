package network

import "fmt"

func (l *Layer) test(testSet []Input) {
	fmt.Println("-------")
	for i := 0; i < len(testSet); i++ {
		l.feedForward(testSet[i].x)
		l.outputFunc()
		fmt.Println("Target: ", testSet[i].y, ", Prediction: ", average(l.output))
	}
	l.feedForward([]float64{2.5})
	l.outputFunc()
	fmt.Println("2.5 prediction: ", average(l.output))

	l.feedForward([]float64{2.3})
	l.outputFunc()
	fmt.Println("2.3 prediction: ", average(l.output))

	l.feedForward([]float64{2.6})
	l.outputFunc()
	fmt.Println("2.6 prediction: ", average(l.output))

	l.feedForward([]float64{8.5})
	l.outputFunc()
	fmt.Println("8.5 prediction: ", average(l.output))
}
