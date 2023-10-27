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
	for i := 0; i < len(data)*100; i++ {
		index := int(math.Floor(rand.Float64() * float64(len(data))))

		index2 := int(math.Floor(rand.Float64() * float64(len(data))))
		original := data[index2]
		random := data[index]
		data[index2] = random
		data[index] = original
	}
}

func loadIrisData() ([][]float64, [][]float64) {
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

	return createIrisData(data[:5])
}

func createIrisData(data [][]string) ([][]float64, [][]float64) {
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
							var _value float64
							switch labelIndex {
							case 0:
								_value = -1
							case 1:
								_value = 0
							case 2:
								_value = 1
							}
							yValues = append(yValues, []float64{_value})
						}
					}
				}

			}

			xValues = append(xValues, _xValues)
		}
	}
	return xValues, yValues
}

func runEpochs(epochs int, function func(int)) {
	for i := 0; i < epochs; i++ {
		function(i)
	}
}
