package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	repoOwner = "itsTurdle"
	repoName  = "llmash"
)

func repoSlug() string {
	if s := env("LLMASH_REPO"); s != "" {
		return s
	}
	return repoOwner + "/" + repoName
}

type release struct {
	Tag    string `json:"tag_name"`
	Draft  bool   `json:"draft"`
	Assets []struct {
		Name string `json:"name"`
		URL  string `json:"browser_download_url"`
	} `json:"assets"`
}

// Tags read v.<main>.<feature>.<patch>-alpha; VERSION holds the number alone.
func (r release) version() string {
	v := strings.TrimPrefix(strings.TrimPrefix(r.Tag, "v"), ".")
	if i := strings.IndexByte(v, '-'); i > 0 {
		v = v[:i]
	}
	return v
}

func latestRelease() (release, error) {
	var rel release
	req, err := http.NewRequest("GET",
		"https://api.github.com/repos/"+repoSlug()+"/releases/latest", nil)
	if err != nil {
		return rel, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "llmash")
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return rel, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	switch resp.StatusCode {
	case 200:
	case 404:
		return rel, fmt.Errorf("%s has no releases yet", repoSlug())
	case 403:
		return rel, fmt.Errorf("GitHub is rate limiting this address; try again later")
	default:
		return rel, fmt.Errorf("GitHub answered %d", resp.StatusCode)
	}
	if err := json.Unmarshal(body, &rel); err != nil {
		return rel, err
	}
	return rel, nil
}

func download(url, dest string) error {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "llmash")
	resp, err := (&http.Client{Timeout: 30 * time.Minute}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("%s: HTTP %d", url, resp.StatusCode)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, throttle(resp.Body))
	return err
}

func cmdUpdate(force bool) {
	release, held := takeUpdateLock()
	if !held {
		die("another %s update is already running on this machine", prog)
	}
	defer release()

	if fileExists(filepath.Join(root, "build.py")) {
		fmt.Printf("this is a source checkout at %s, not an install.\n"+
			"Update it with:  python build.py --here\n", root)
		exit(1)
	}

	rel, err := latestRelease()
	if err != nil {
		die("could not reach GitHub: %v", err)
	}
	here, there := versionString(), rel.version()
	fmt.Printf("installed %s, %s has %s\n", here, repoSlug(), there)
	if here == there && !force {
		fmt.Println("already up to date")
		return
	}

	var script string
	for _, a := range rel.Assets {
		if a.Name == "install.ps1" {
			script = a.URL
		}
	}
	if script == "" {
		script = "https://raw.githubusercontent.com/" + repoSlug() + "/main/install.ps1"
	}

	tmp := filepath.Join(os.TempDir(), fmt.Sprintf("llmash-update-%d.ps1", os.Getpid()))
	if err := download(script, tmp); err != nil {
		die("could not fetch the installer: %v", err)
	}
	defer os.Remove(tmp)

	fmt.Printf("updating %s to %s\n\n", root, there)
	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
		"-File", tmp, "-Dir", root, "-Yes")
	cmd.Stdout, cmd.Stderr, cmd.Stdin = os.Stdout, os.Stderr, os.Stdin
	if err := cmd.Run(); err != nil {
		die("the installer stopped: %v", err)
	}
}
