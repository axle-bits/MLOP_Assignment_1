# Local container proof (2026-07-04)

## Build command
```bash
docker build -t heart-disease-api -f infra/Dockerfile .
```

## Build output (last 5 lines)
```
#14 exporting manifest list sha256:18069d8a0ef3d61cf0b18a7645d0234adbe7a4410f6db20a4a94d2cd9b142a2e 0.0s done
#14 naming to docker.io/library/heart-disease-api:latest done
#14 unpacking to docker.io/library/heart-disease-api:latest
#14 unpacking to docker.io/library/heart-disease-api:latest 2.9s done
#14 DONE 10.5s
```

## Run command
```bash
docker run -d --name heart-api -p 8000:8000 heart-disease-api
```

## Container ID
```
d87746adc0ad14e1d4c63b646e92fbc1ba9f432b73b86f3d256c88959dfdfa1f
```

## Health endpoint (curl -s http://localhost:8000/health)
```json
{"status":"ok","model_loaded":true}
```

## Predict endpoint (curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @api/sample_request.json)
```json
{"prediction":0,"probability":0.3326011601407892,"risk_label":"no disease"}
```

## Docker logs (last 5 lines)
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:50064 - "GET /health HTTP/1.1" 200 OK
INFO:     172.17.0.1:46556 - "GET /health HTTP/1.1" 200 OK
2026-07-04 09:13:00,144 heart_disease_api INFO predict prediction=0 probability=0.3326 latency_ms=14.4
INFO:     172.17.0.1:46560 - "POST /predict HTTP/1.1" 200 OK
```
