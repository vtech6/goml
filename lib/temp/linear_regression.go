package temp

import "fmt"

type LinearInput struct {
	x int
	y int
}

func LinearRegression() {
	fmt.Println("HELLO GO")
	custominput := [3]LinearInput{
		{x: 1, y: 3}, {x: 2, y: 5}, {x: 3, y: 7},
	}
	testInput := [2]LinearInput{{x: 8, y: 17}, {x: 9, y: 19}}
	testInput2 := [2]LinearInput{{x: 3, y: 29}, {x: 5, y: 43}}

	inputs := [][]LinearInput{custominput[:], testInput[:]}
	for i := range inputs {
		fmt.Println(calculateCofactor(inputs[i]))
	}
	getMultiplier(testInput2[:])
}

func calculateCofactor(input []LinearInput) int {
	cofactor := 0
	for i := range input {
		value := input[i]
		cofactor += (value.y - value.x)
	}
	if len(input) > 1 {
		cofactor = cofactor / (len(input) - 1)
	}
	return cofactor
}

func getMultiplier(input []LinearInput) (int, int) {
	if len(input) > 1 {
		x1 := input[0].x
		x2 := input[1].x
		y1 := input[0].y
		y2 := input[1].y

		n := (y2 - y1) / (x2 - x1)
		b := y1 - (n * x1)

		fmt.Printf("%d, %d", n, b)
		return n, b
	}
	return 0, 0
}
