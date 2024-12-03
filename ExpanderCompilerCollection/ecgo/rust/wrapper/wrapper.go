package wrapper

/*
#include <stdlib.h>
#include <dlfcn.h>
#include "./wrapper.h"
*/
import "C"
import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/consensys/gnark/logger"
)

const ABI_VERSION = 4

func getCacheDir() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	cacheDir := filepath.Join(homeDir, ".cache", "ExpanderCompilerCollection")
	err = os.MkdirAll(cacheDir, 0755)
	return cacheDir, err
}

var compilePtr unsafe.Pointer = nil
var proveCircuitFilePtr unsafe.Pointer = nil
var verifyCircuitFilePtr unsafe.Pointer = nil
var compilePtrLock sync.Mutex

func downloadFile(url string, filepath string) error {
	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return err
	}

	return nil
}

func getUrl(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bad status: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return body, nil
}

func getLibName() string {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return "libec_go_lib.dylib"
		}
	case "linux":
		if runtime.GOARCH == "amd64" {
			return "libec_go_lib.so"
		}
	}
	panic(fmt.Sprintf("unsupported platform %s %s", runtime.GOOS, runtime.GOARCH))
}

func downloadLib(path string) {
	log := logger.Logger()
	log.Info().Msg("Downloading rust libs ...")
	err := downloadFile("https://github.com/PolyhedraZK/ExpanderCompilerCollection/raw/rust-built-libs/"+getLibName(), path)
	if err != nil {
		os.Remove(path)
		panic(err)
	}
}

type repoInfo struct {
	Commit struct {
		Commit struct {
			Committer struct {
				Date string `json:"date"`
			} `json:"committer"`
		} `json:"commit"`
	} `json:"commit"`
}

func updateLib(path string) {
	stat, err := os.Stat(path)
	fileExists := !os.IsNotExist(err)
	if err != nil && fileExists {
		panic(err)
	}
	data, err := getUrl("https://api.github.com/repos/PolyhedraZK/ExpanderCompilerCollection/branches/rust-built-libs")
	if err != nil {
		if fileExists {
			return
		}
		panic(err)
	}
	var repoInfo repoInfo
	err = json.Unmarshal(data, &repoInfo)
	if err != nil {
		if fileExists {
			return
		}
		panic(err)
	}
	remoteTime, err := time.Parse(time.RFC3339, repoInfo.Commit.Commit.Committer.Date)
	if err != nil {
		if fileExists {
			return
		}
		panic(err)
	}
	if fileExists {
		localTime := stat.ModTime()
		if localTime.After(remoteTime) {
			return
		}
	}
	downloadLib(path)
}

func initCompilePtr() {
	compilePtrLock.Lock()
	defer compilePtrLock.Unlock()
	if compilePtr != nil {
		return
	}
	cacheDir, err := getCacheDir()
	if err != nil {
		panic(fmt.Sprintf("failed to get cache dir: %v", err))
	}
	soPath := filepath.Join(cacheDir, getLibName())
	updateLib(soPath)
	handle := C.dlopen(C.CString(soPath), C.RTLD_LAZY)
	if handle == nil {
		panic("failed to load libec_go_lib, you may need to install openmpi")
	}
	abiVersionPtr := C.dlsym(handle, C.CString("abi_version"))
	if abiVersionPtr == nil {
		panic("failed to load abi_version function")
	}
	abiVersion := C.abi_version(abiVersionPtr)
	if abiVersion != ABI_VERSION {
		panic("abi_version mismatch, please consider update the go package")
	}
	compilePtr = C.dlsym(handle, C.CString("compile"))
	if compilePtr == nil {
		panic("failed to load compile function")
	}
	proveCircuitFilePtr = C.dlsym(handle, C.CString("prove_circuit_file"))
	if proveCircuitFilePtr == nil {
		panic("failed to load prove_circuit_file function")
	}
	verifyCircuitFilePtr = C.dlsym(handle, C.CString("verify_circuit_file"))
	if verifyCircuitFilePtr == nil {
		panic("failed to load verify_circuit_file function")
	}
}

// from c to go
func goBytes(data *C.uint8_t, length C.uint64_t) []byte {
	return bytes.Clone(unsafe.Slice((*byte)(data), length))
}

func CompileWithRustLib(s []byte, configId uint64) ([]byte, []byte, error) {
	initCompilePtr()

	in := C.ByteArray{data: (*C.uint8_t)(C.CBytes(s)), length: C.uint64_t(len(s))}
	defer C.free(unsafe.Pointer(in.data))

	cr := C.compile(compilePtr, in, C.uint64_t(configId))

	defer C.free(unsafe.Pointer(cr.ir_witness_gen.data))
	defer C.free(unsafe.Pointer(cr.layered.data))
	defer C.free(unsafe.Pointer(cr.error.data))

	irWitnessGen := goBytes(cr.ir_witness_gen.data, cr.ir_witness_gen.length)
	layered := goBytes(cr.layered.data, cr.layered.length)
	errMsg := goBytes(cr.error.data, cr.error.length)

	if len(errMsg) > 0 {
		return nil, nil, errors.New(string(errMsg))
	}

	return irWitnessGen, layered, nil
}

func ProveCircuitFile(circuitFilename string, witness []byte, configId uint64) []byte {
	initCompilePtr()
	bytesFn := []byte(circuitFilename)
	cf := C.ByteArray{data: (*C.uint8_t)(C.CBytes(bytesFn)), length: C.uint64_t(len(bytesFn))}
	defer C.free(unsafe.Pointer(cf.data))
	wi := C.ByteArray{data: (*C.uint8_t)(C.CBytes(witness)), length: C.uint64_t(len(witness))}
	defer C.free(unsafe.Pointer(wi.data))
	proof := C.prove_circuit_file(proveCircuitFilePtr, cf, wi, C.uint64_t(configId))
	defer C.free(unsafe.Pointer(proof.data))
	return goBytes(proof.data, proof.length)
}

func VerifyCircuitFile(circuitFilename string, witness []byte, proof []byte, configId uint64) bool {
	initCompilePtr()
	bytesFn := []byte(circuitFilename)
	cf := C.ByteArray{data: (*C.uint8_t)(C.CBytes(bytesFn)), length: C.uint64_t(len(bytesFn))}
	defer C.free(unsafe.Pointer(cf.data))
	wi := C.ByteArray{data: (*C.uint8_t)(C.CBytes(witness)), length: C.uint64_t(len(witness))}
	defer C.free(unsafe.Pointer(wi.data))
	pr := C.ByteArray{data: (*C.uint8_t)(C.CBytes(proof)), length: C.uint64_t(len(proof))}
	defer C.free(unsafe.Pointer(pr.data))
	return C.verify_circuit_file(verifyCircuitFilePtr, cf, wi, pr, C.uint64_t(configId)) != 0
}
