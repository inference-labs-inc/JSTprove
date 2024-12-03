package utils

import "math/bits"

// pad to 2^n gates (and 4^n for first layer)
// 4^n exists for historical reasons, not used now
func NextPowerOfTwo(x int, is4 bool) int {
	if x < 0 {
		panic("x must be non-negative")
	}

	padk := bits.Len(uint(x))
	if is4 && padk%2 != 0 {
		padk++
	}
	return 1 << padk
}
