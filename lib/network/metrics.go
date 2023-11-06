package network

import (
	"fmt"
	"math"
)

func bceMetrics(network MLP, testX [][]float64, minValue float64, minMaxScale float64, testY [][]float64) float64 {
	accuracy := 0.0
	for i := range testX {
		val := network.calculateOutput(testX[i])
		outputValue := math.Round((val[0].value - minValue) / minMaxScale)
		targetValue := testY[i][0]
		fmt.Println("Expected output:", targetValue, "Prediction:", outputValue)
		if outputValue == targetValue {
			accuracy += 1
		}
	}

	accuracy = float64(accuracy) / float64(len(testY))
	accuracy = math.Round(accuracy * 100)
	return accuracy
}

func regressionMetrics(network MLP, testX [][]float64, minValue float64, minMaxScale float64, testY [][]float64) float64 {
	accuracy := 0.0
	for i := range testX {
		val := network.calculateOutput(testX[i])
		outputValue := math.Round(((val[0].value-minValue)/minMaxScale)*2.0) - 1
		targetValue := testY[i][0]
		fmt.Println("Expected output:", targetValue, "Prediction:", outputValue)
		if outputValue == targetValue {
			accuracy += 1
		}
	}

	accuracy = float64(accuracy) / float64(len(testY))
	accuracy = math.Round(accuracy * 100)
	return accuracy
}