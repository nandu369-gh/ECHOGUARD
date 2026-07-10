package config

import "os"

type Config struct {
	Port      string
	ModelPath string
}

func LoadConfig() *Config {
	// Look for a custom system port, otherwise fall back to standard 8080
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Look for custom model targets, otherwise default to our newly generated file location
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "models/audio_classifier.onnx"
	}

	return &Config{Port: port, ModelPath: modelPath}
}
