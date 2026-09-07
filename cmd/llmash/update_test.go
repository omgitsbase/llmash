package main

import "testing"

// Releases are tagged v.<main>.<feature>.<patch>-alpha but VERSION holds the
// number alone, so the two have to meet. Getting this wrong means update either
// never fires or fires forever.
func TestReleaseVersion(t *testing.T) {
	cases := map[string]string{
		"v.0.2.1-alpha": "0.2.1",
		"v.0.2.1":       "0.2.1",
		"v0.2.1":        "0.2.1",
		"0.2.1":         "0.2.1",
		"v.1.0.0-beta":  "1.0.0",
	}
	for tag, want := range cases {
		if got := (release{Tag: tag}).version(); got != want {
			t.Errorf("tag %q read as %q, want %q", tag, got, want)
		}
	}
}

// The installed version and the tag of the release built from it must compare
// equal, or `llmash update` offers an update to the version already installed.
func TestInstalledMatchesItsOwnTag(t *testing.T) {
	if got := (release{Tag: "v." + serverVersion + "-alpha"}).version(); got != serverVersion {
		t.Errorf("this build is %s but its own tag reads as %s", serverVersion, got)
	}
}
