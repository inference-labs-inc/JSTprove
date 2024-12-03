package utils

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

type SourceInfo struct {
	File string
	Line int
}

func ShowProfiling(varSourceInfo []SourceInfo, varCost []int) {
	fileCost := make(map[string]map[int]int)
	for i := 0; i < len(varSourceInfo) && i < len(varCost); i++ {
		if fileCost[varSourceInfo[i].File] == nil {
			fileCost[varSourceInfo[i].File] = make(map[int]int)
		}
		fileCost[varSourceInfo[i].File][varSourceInfo[i].Line] += varCost[i]
	}
	files := []string{}
	for file := range fileCost {
		files = append(files, file)
	}
	sort.Strings(files)

	for _, file := range files {
		if file == "finalize" || file == "input" || file == "one" {
			continue
		}
		totalCost := 0
		lines := []int{}
		for line, cost := range fileCost[file] {
			if cost != 0 {
				lines = append(lines, line)
			}
			totalCost += cost
		}
		sort.Ints(lines)
		fmt.Printf("File: %s | Total Cost: %d\n", file, totalCost)
		content, err := os.ReadFile(file)
		if err != nil {
			fmt.Printf("Can't read file content\n\n")
			continue
		}
		fileLines := strings.Split(string(content), "\n")

		fmt.Printf("%7s | Line | Code\n", "Cost")
		lastLine := 0
		for _, line := range lines {
			for i := line - 2; i <= line+2; i++ {
				if lastLine >= i {
					continue
				}
				if i-lastLine > 1 {
					fmt.Printf("%7s | %4s | ...\n", "", "")
				}
				lastLine = i
				cost := fileCost[file][i]
				if cost != 0 {
					fmt.Printf("%7d | %4d | %s\n", cost, i, fileLines[i-1])
				} else {
					fmt.Printf("%7s | %4d | %s\n", "", i, fileLines[i-1])
				}
			}
		}
		fmt.Println()
	}
}
