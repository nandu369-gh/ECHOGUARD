# 🛡️ EchoGuard Serving Layer

EchoGuard is a high-performance, low-latency machine learning inference service built in Go. It wraps an optimized ONNX Runtime execution engine to analyze structural token/acoustic embeddings and provide safety classification in real time via a single standalone file.

---

## 🚀 Quick Start (Local macOS Development)

### 1. Install System Dependencies
EchoGuard utilizes native bindings to interface with the C++ ONNX Runtime. Install the library via Homebrew:
```bash
brew install onnxruntime
```
*This installs `libonnxruntime.dylib` to your native system path (`/opt/homebrew/lib/`).*

### 2. Workspace Setup
Initialize your Go environment and download the required modules:
```bash
# Clean up and reset tracking states if needed
rm -f go.mod go.sum

# Initialize and pull components
go mod init echoguard
go get ://github.com
go get ://github.com
go mod tidy
```

### 3. Add Your Model
Ensure your compiled machine learning model file is saved directly into the root workspace directory and named exactly:
```text
model.onnx
```

### 4. Run the Engine
Boot up the HTTP serving engine:
```bash
go run main.go
```
*Console output will confirm initialization:* `EchoGuard Serving Layer starting on port 8080...`

---

## 📡 API Specification

### Inference Moderation Endpoint

* **URL Path:** `/v1/moderation`
* **HTTP Method:** `POST`
* **Headers:** `Content-Type: application/json`

#### Request Payload Schema
The engine expects a fixed-dimension vector embedding containing exactly 128 elements.

```json
{
  "features": [0.1, 0.2, -0.5, 0.0, 1.2, 0.9, -0.1, 0.4, ... 128 elements total ...]
}
```

#### Response Payload Schema

| Property | Data Type | Value Scopes / Description |
| :--- | :--- | :--- |
| `status` | `String` | `"success"` or `"error"` |
| `class` | `String` | `"Safe"`, `"Toxic"`, or `"Urgent Threat"` |
| `code` | `Integer` | `0` (Safe), `1` (Toxic), `2` (Urgent Threat) |

**Example Successful Response (HTTP 200 OK):**
```json
{
  "status": "success",
  "class": "Toxic",
  "code": 1
}
```

#### Verification Curl Request
Open an independent terminal window and run the following command to test the API endpoint pipeline:

```bash
curl -X POST http://localhost:8080/v1/moderation \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4,0.3,0.2,0.1,0.0,0.1,0.2,-0.5,0.0,1.2,0.9,-0.1,0.4]}'
```

---

## 🐳 Docker Deployment (Production)

Production environments run on a Linux footprint. The application handles architecture mismatches inside the container by automatically downloading the matching shared library version of Linux `libonnxruntime.so`.

### Build Image
When compiling on an Apple Silicon Mac for an x86 Linux production cloud host, specify the target architecture platform flag explicitly:
```bash
docker build --platform linux/amd64 -t echoguard-serving:latest .
```

### Run Container
```bash
docker run -d -p 8080:8080 --name echoguard-service echoguard-serving:latest
```
