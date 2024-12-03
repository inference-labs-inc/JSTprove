package utils

const CostOfInput = 1000
const CostOfVariable = 100
const CostOfMulGate = 10
const CostOfAddGate = 3
const CostOfCstGate = 3

func CostOfMultiply(aDeg0, aDeg1, bDeg0, bDeg1 int) int {
	return CostOfMulGate*(aDeg1*bDeg1) + CostOfAddGate*(aDeg0*bDeg1+aDeg1*bDeg0) + CostOfCstGate*(aDeg0*bDeg0)
}

func CostOfCompress(deg0, deg1, deg2 int) int {
	return CostOfMulGate*deg2 + CostOfAddGate*deg1 + CostOfCstGate*deg0 + CostOfVariable
}
