package inference

import (
	"fmt"
	"math"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

type Engine struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
}

func NewEngine(modelPath string) (*Engine, error) {
	// Dynamically pick the right file path based on the operating system
	if runtime.GOOS == "linux" {
		// Standard container dynamic runtime tracking target path configuration
		ort.SetSharedLibraryPath("libonnxruntime.so")
	} else {
		// Local development footprint on your Mac MacBook Air
		ort.SetSharedLibraryPath("/opt/homebrew/lib/libonnxruntime.dylib")
	}

	// Initialize runtime environment
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to init runtime: %w", err)
	}

	// Allocate backing tensor memory structures
	inputShape := ort.NewShape(1, 128)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("failed creating static input template: %w", err)
	}

	outputShape := ort.NewShape(1, 3)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("failed creating static output template: %w", err)
	}

	// Instantiate the session by explicitly pre-binding your tensor configurations
	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input_audio"},
		[]string{"predictions"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		outputTensor.Destroy()
		inputTensor.Destroy()
		ort.DestroyEnvironment()
		return nil, fmt.Errorf("failed to load model session framework: %w", err)
	}

	return &Engine{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

func (e *Engine) Predict(features []float32) (int, error) {
	// Safely copy incoming network vectors into our pre-bound input memory slice
	inputSlice := e.inputTensor.GetData()
	copy(inputSlice, features)

	// Run takes zero arguments because the arrays are pre-bound at startup
	err := e.session.Run()
	if err != nil {
		return 0, fmt.Errorf("onnx engine prediction run cycle failed: %w", err)
	}

	// Extract the model classification metrics
	outputData := e.outputTensor.GetData()

	// Argmax routing logic to fetch the target category index
	maxIdx := 0
	maxVal := float32(-math.MaxFloat32)
	for i, val := range outputData {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx, nil
}

func (e *Engine) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
	if e.outputTensor != nil {
		e.outputTensor.Destroy()
	}
	if e.inputTensor != nil {
		e.inputTensor.Destroy()
	}
	ort.DestroyEnvironment()
}
