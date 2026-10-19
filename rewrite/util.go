package rewrite

import "sort"

func product(vs []int64) int64 {
	p := int64(1)
	for _, v := range vs {
		p *= v
	}
	return p
}

func ones(n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func anyNonZero(vs []int64) bool {
	for _, v := range vs {
		if v != 0 {
			return true
		}
	}
	return false
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
