package integration

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"runtime"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/rust"
)

type Prover struct {
	circuitDir string
	filename   string
}

func is_url(s string) bool {
	_, err := url.ParseRequestURI(s)
	return err == nil
}

// for compatibility with keccak_serve
func GetIntegrationBinLoc() string {
	switch runtime.GOOS {
	case "darwin":
		return "./integration/bin/expander-exec-macos"
	case "linux":
		return "./integration/bin/expander-exec-linux-avx2"
	default:
		println("Unsupported OS: ", runtime.GOOS)
		panic("Unsupported OS")
	}
}

// format from external specification, maxConcurrency and releaseFlags are not used now
func NewProver(circuitDir string, filename string, maxConcurrency int, releaseFlag bool) (*Prover, error) {
	return &Prover{
		circuitDir: circuitDir,
		filename:   filename,
	}, nil
}

func (p *Prover) Prove(witnessData []byte) ([]byte, error) {
	if is_url(p.filename) {
		proveReq, err := http.NewRequest("POST", p.filename+"/prove", bytes.NewBuffer(witnessData))
		if err != nil {
			return nil, err
		}
		proveReq.Header.Set("Content-Type", "application/octet-stream")
		proveReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(witnessData)))
		client := &http.Client{}
		proveResp, err := client.Do(proveReq)
		if err != nil {
			return nil, err
		}
		defer proveResp.Body.Close()

		proof_and_claim, err := io.ReadAll(proveResp.Body)
		if err != nil {
			return nil, err
		}
		return proof_and_claim, nil
	}
	return rust.ProveFile(p.circuitDir, witnessData), nil
}

func (p *Prover) Verify(witnessData []byte, proof_and_claim []byte) (bool, error) {
	if is_url(p.filename) {
		witnessLen := make([]byte, 8)
		binary.LittleEndian.PutUint64(witnessLen, uint64(len(witnessData)))
		proofLen := make([]byte, 8)
		binary.LittleEndian.PutUint64(proofLen, uint64(len(proof_and_claim)))
		verifierInput := append(append(append(witnessLen, proofLen...), witnessData...), proof_and_claim...)
		verifyReq, err := http.NewRequest("POST", p.filename+"/verify", bytes.NewBuffer(verifierInput))
		if err != nil {
			return false, err
		}
		verifyReq.Header.Set("Content-Type", "application/octet-stream")
		verifyReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(verifierInput)))
		client := &http.Client{}
		verifyResp, err := client.Do(verifyReq)
		if err != nil {
			return false, err
		}
		defer verifyResp.Body.Close()

		verificationResult, err := io.ReadAll(verifyResp.Body)
		if err != nil {
			return false, err
		}
		return string(verificationResult) == "success", nil
	}
	return rust.VerifyFile(p.circuitDir, witnessData, proof_and_claim), nil
}
