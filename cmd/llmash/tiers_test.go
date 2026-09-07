package main

import "testing"

func TestTiers(t *testing.T) {
	q := []quantInfo{{"UD-IQ2_M", 2290860128, 1}, {"Q3_K_S", 2445652064, 1}, {"Q3_K_M", 2536786016, 1}, {"IQ4_XS", 2983944288, 1},
		{"Q4_0", 3041378400, 1}, {"Q4_K_M", 3106738272, 1}, {"Q5_K_M", 3356037216, 1}, {"Q6_K", 4501721184, 1}, {"Q8_0", 5048352864, 1}, {"BF16", 9311305568, 1}}
	tr := tiersOf(q)
	if q[tr.tiny].Name != "Q3_K_M" || q[tr.medium].Name != "Q4_K_M" || q[tr.large].Name != "Q8_0" {
		t.Fatalf("got %s / %s / %s", q[tr.tiny].Name, q[tr.medium].Name, q[tr.large].Name)
	}
	q2 := []quantInfo{{"Q4_0", 2841481184, 1}, {"Q8_0", 4967497152, 1}, {"BF16", 9311305152, 1}}
	tr = tiersOf(q2)
	if q2[tr.tiny].Name != "Q4_0" || q2[tr.medium].Name != "Q4_0" || q2[tr.large].Name != "Q8_0" {
		t.Fatalf("got %s / %s / %s", q2[tr.tiny].Name, q2[tr.medium].Name, q2[tr.large].Name)
	}
}
