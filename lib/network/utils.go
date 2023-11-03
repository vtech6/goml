package network

import (
	"encoding/csv"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
)

const (
	TestMode = false
)

func randomFloatStd() float64 {
	if TestMode == true {
		return 1
	} else {
		return (rand.Float64() * 2) - 1
	}
}

func generateBias() float64 {
	if TestMode == true {

		return 0
	} else {
		return randomFloatStd()
	}
}

func shuffleData(data [][]string) {
	for times := 0; times < 10; times++ {
		for i := len(data) - 1; i > 0; i-- {
			j := int(math.Floor(rand.Float64() * float64(i+1)))
			original := data[i]
			random := data[j]
			data[i] = random
			data[j] = original
		}
	}
}

func loadIrisData(trainTestSplitRatio float64, targetLabels [][]float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	file, error := os.Open("./lib/dataset/IRIS.csv")
	if error != nil {
		log.Fatal(error)
	}
	defer file.Close()

	csvReader := csv.NewReader(file)
	data, error := csvReader.ReadAll()
	if error != nil {
		log.Fatal(error)
	}
	shuffleData(data[1:])
	x, y := createIrisData(data, targetLabels)

	return trainTestSplit(x, y, trainTestSplitRatio)
}

func createIrisData(data [][]string, targetLabels [][]float64) ([][]float64, [][]float64) {
	labels := []string{"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
	xValues := make([][]float64, 0)
	yValues := make([][]float64, 0)
	for i, line := range data {
		if i > 0 {
			_xValues := make([]float64, 0)
			for j, entry := range line {
				if j < 4 {
					value, error := strconv.ParseFloat(entry, 64)
					if error != nil {
						log.Fatal(error)
					}

					_xValues = append(_xValues, value)
				}
				if j == 4 {
					for labelIndex, label := range labels {
						if label == entry {
							var _value []float64
							switch labelIndex {
							case 0:
								_value = targetLabels[0]
							case 1:
								_value = targetLabels[1]
							case 2:
								_value = targetLabels[2]
							}
							yValues = append(yValues, _value)
						}
					}
				}

			}

			xValues = append(xValues, _xValues)
		}
	}
	minMaxData(xValues)
	return xValues, yValues
}

func runEpochs(epochs int, function func(int)) {
	for i := 0; i < epochs; i++ {
		function(i)
	}
}

func minMaxData(data [][]float64) {
	for i := range data[0] {
		minVal := 5.0
		maxVal := 0.0
		for j := range data {
			if data[j][i] < minVal {
				minVal = data[j][i]
			}
			if data[j][i] > maxVal {
				maxVal = data[j][i]
			}

		}
		for j := range data {
			data[j][i] = (data[j][i] - minVal) / (maxVal - minVal)
		}
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

func trainTestSplit(dataX [][]float64, dataY [][]float64, ratio float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	splitIndex := int(math.Floor(float64(len(dataX)-1) * ratio))
	return dataX[:splitIndex], dataY[:splitIndex], dataX[splitIndex:], dataY[splitIndex:]
}
