package main

import (
	"log"

	"echoguard/internal/api"
	"echoguard/internal/config"
	"echoguard/internal/inference"

	"github.com/gin-gonic/gin"
)

func main() {
	// 1. Ingest operating system environments and configurations
	cfg := config.LoadConfig()

	// 2. Initialize the underlying machine learning model engine bindings
	eng, err := inference.NewEngine(cfg.ModelPath)
	if err != nil {
		log.Fatalf("Critical error initializing inference core framework: %v", err)
	}
	defer eng.Close()

	// 3. Launch the high-speed web frame routing engine
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())

	// 4. Mount API endpoint route triggers to the active server pipeline
	api.RegisterRoutes(r, eng)

	log.Printf("EchoGuard Serving Layer starting on port %s...", cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatalf("Server stopped unexpectedly: %v", err)
	}
}
