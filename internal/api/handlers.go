package api

import (
	"echoguard/internal/inference"
	"net/http"

	"github.com/gin-gonic/gin"
)

// AudioRequest specifies the layout required for incoming payloads
type AudioRequest struct {
	Features []float32 `json:"features" binding:"required"` // Must contain exactly 128 elements
}

// RegisterRoutes configures endpoints and links them to our ML engine
func RegisterRoutes(r *gin.Engine, eng *inference.Engine) {
	r.POST("/v1/moderation", func(c *gin.Context) {
		var req AudioRequest

		// Validate JSON structure
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid features payload format structure"})
			return
		}

		// Ensure raw data array elements match our model properties exactly
		if len(req.Features) != 128 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Features array data payload dimensions must equal 128"})
			return
		}

		// Push data points into our machine learning model inference pipeline
		labelIdx, err := eng.Predict(req.Features)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Machine Learning inference model pass pipeline failed"})
			return
		}

		// Map index integers back to standard readable classifications
		labels := []string{"Safe", "Toxic", "Urgent Threat"}
		c.JSON(http.StatusOK, gin.H{
			"status": "success",
			"class":  labels[labelIdx],
			"code":   labelIdx,
		})
	})
}
