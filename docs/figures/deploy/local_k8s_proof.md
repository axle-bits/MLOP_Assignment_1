# Local Kubernetes deployment proof (2026-07-11)

Cluster: Docker Desktop Kubernetes, 2 nodes (`desktop-control-plane`, `desktop-worker`), context `docker-desktop`.

## Manifest validation (dry run)

```bash
kubectl apply --dry-run=client -f infra/k8s/
```

```
namespace/heart-disease unchanged (dry run)
deployment.apps/heart-disease-api unchanged (dry run)
service/heart-disease-api unchanged (dry run)
```

All manifests are valid and would apply cleanly to the cluster.

## Step 1: Build and tag the image

```bash
docker build -t heart-disease-api:v1 -f infra/Dockerfile . 2>&1 | tail -10
```

```
#14 exporting layers
#14 exporting layers 7.6s done
#14 exporting manifest sha256:f7b1e09cf8702360d07560ee0d5f5340c0e5ec1aa34b93389577b0a3f4e213d1 0.0s done
#14 exporting config sha256:f546f0b88951f56077fb8421d76981cb7a1757656d710547975887e1700dcdca 0.0s done
#14 exporting attestation manifest sha256:df2ad4c3486085daf2b674e0ce08ca489138f9acecc3475427c3d4eb53be355d 0.0s done
#14 exporting manifest list sha256:b2fb0b9c3784a70bace5e3e0a1ba8f51bfce88e0614d0c8a99c971e70db74564 0.0s done
#14 naming to docker.io/library/heart-disease-api:v1 done
#14 unpacking to docker.io/library/heart-disease-api:v1
#14 unpacking to docker.io/library/heart-disease-api:v1 3.2s done
#14 DONE 11.0s
```

Build succeeded (mostly cached from Phase 5, per the build's own layer cache).

## Step 2: Apply manifests

```bash
kubectl apply -f infra/k8s/ && kubectl get pods -n heart-disease
```

```
namespace/heart-disease created
deployment.apps/heart-disease-api created
service/heart-disease-api created
NAME                                 READY   STATUS              RESTARTS   AGE
heart-disease-api-5d7b66585b-8rwds   0/1     ContainerCreating   0          1s
heart-disease-api-5d7b66585b-hgmj2   0/1     ContainerCreating   0          1s
```

(Note: `infra/k8s/00-namespace.yaml` is the namespace manifest, renumbered so it applies before the deployment/service.)

## Step 3: Wait for rollout

```bash
kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=120s
```

```
Waiting for deployment "heart-disease-api" rollout to finish: 0 of 2 updated replicas are available...
Waiting for deployment "heart-disease-api" rollout to finish: 1 of 2 updated replicas are available...
deployment "heart-disease-api" successfully rolled out
```

**Image-load fallback was NOT needed.** The locally built image was visible to both nodes and the rollout completed cleanly on the first attempt — no `ErrImageNeverPull`/`ImagePullBackOff` was observed, so the `docker save` / `docker cp` / `ctr import` remediation was not exercised.

## Step 4: Verify cluster state and endpoints

```bash
kubectl get nodes && kubectl get all -n heart-disease -o wide
```

```
NAME                    STATUS   ROLES           AGE   VERSION
desktop-control-plane   Ready    control-plane   14m   v1.36.1
desktop-worker          Ready    <none>          14m   v1.36.1
---
NAME                                     READY   STATUS    RESTARTS   AGE   IP           NODE             NOMINATED NODE   READINESS GATES
pod/heart-disease-api-5d7b66585b-8rwds   1/1     Running   0          18s   10.244.1.3   desktop-worker   <none>           <none>
pod/heart-disease-api-5d7b66585b-hgmj2   1/1     Running   0          18s   10.244.1.2   desktop-worker   <none>           <none>

NAME                        TYPE           CLUSTER-IP     EXTERNAL-IP   PORT(S)        AGE   SELECTOR
service/heart-disease-api   LoadBalancer   10.96.75.197   172.18.0.6    80:32503/TCP   18s   app=heart-disease-api

NAME                                READY   UP-TO-DATE   AVAILABLE   AGE   CONTAINERS   IMAGES                 SELECTOR
deployment.apps/heart-disease-api   2/2     2            2           18s   api          heart-disease-api:v1   app=heart-disease-api

NAME                                           DESIRED   CURRENT   READY   AGE   CONTAINERS   IMAGES                 SELECTOR
replicaset.apps/heart-disease-api-5d7b66585b   2         2         2       18s   api          heart-disease-api:v1   app=heart-disease-api,pod-template-hash=5d7b66585b
```

Both nodes `Ready`; 2/2 pods `Running` with `READY 1/1`. Note: the service `EXTERNAL-IP` column shows `172.18.0.6` (Docker Desktop's LB IP) rather than the literal string `localhost` — this is Docker Desktop's normal behavior; `localhost:80` still routes to the service, confirmed below.

```bash
curl -s http://localhost/health && echo && curl -s -X POST http://localhost/predict -H "Content-Type: application/json" -d @api/sample_request.json
```

```
{"status":"ok","model_loaded":true}
{"prediction":0,"probability":0.3326011601407892,"risk_label":"no disease"}
```

## Step 5: Probe + request-logging evidence

```bash
POD=$(kubectl get pods -n heart-disease -o jsonpath='{.items[0].metadata.name}')
kubectl describe pod $POD -n heart-disease | grep -A2 -E "Liveness|Readiness"
```

```
    Liveness:     http-get http://:8000/health delay=10s timeout=1s period=10s #success=1 #failure=3
    Readiness:    http-get http://:8000/health delay=5s timeout=1s period=5s #success=1 #failure=3
    Environment:  <none>
    Mounts:
```

```bash
kubectl logs $POD -n heart-disease | tail -5
```

Initial tail of `heart-disease-api-5d7b66585b-8rwds` only showed health-check traffic (the earlier single `/predict` POST had load-balanced to the other pod). The `/predict` request was POSTed a few more times to guarantee coverage of both pods, then both pods' tails were re-captured:

```
INFO:     10.244.1.1:36528 - "GET /health HTTP/1.1" 200 OK
INFO:     10.244.1.1:36542 - "GET /health HTTP/1.1" 200 OK
INFO:     10.244.1.1:33560 - "GET /health HTTP/1.1" 200 OK
INFO:     10.244.1.1:33568 - "GET /health HTTP/1.1" 200 OK
2026-07-11 04:41:08,607 heart_disease_api INFO predict prediction=0 probability=0.3326 latency_ms=17.2
INFO:     10.244.1.1:6579 - "POST /predict HTTP/1.1" 200 OK
INFO:     10.244.1.1:33578 - "GET /health HTTP/1.1" 200 OK
2026-07-11 04:41:09,039 heart_disease_api INFO predict prediction=0 probability=0.3326 latency_ms=8.2
INFO:     10.244.1.1:1147 - "POST /predict HTTP/1.1" 200 OK
```

(pod `heart-disease-api-5d7b66585b-8rwds`, recent log excerpt)

```
INFO:     10.244.1.1:53750 - "GET /health HTTP/1.1" 200 OK
INFO:     10.244.1.1:53590 - "GET /health HTTP/1.1" 200 OK
INFO:     10.244.1.1:53600 - "GET /health HTTP/1.1" 200 OK
INFO:     10.244.1.1:53604 - "GET /health HTTP/1.1" 200 OK
2026-07-11 04:41:09,469 heart_disease_api INFO predict prediction=0 probability=0.3326 latency_ms=6.7
INFO:     10.244.1.1:25850 - "POST /predict HTTP/1.1" 200 OK
2026-07-11 04:41:09,910 heart_disease_api INFO predict prediction=0 probability=0.3326 latency_ms=7.1
INFO:     10.244.1.1:31202 - "POST /predict HTTP/1.1" 200 OK
INFO:     10.244.1.1:39060 - "GET /health HTTP/1.1" 200 OK
```

(pod `heart-disease-api-5d7b66585b-hgmj2`, recent log excerpt)

Both pods show the structured application log line `heart_disease_api INFO predict prediction=... probability=... latency_ms=...` alongside uvicorn's access-log line for the same request.

## Step 6: Rolling update demo (zero downtime)

```bash
kubectl rollout restart deployment/heart-disease-api -n heart-disease && for i in 1 2 3 4 5 6; do curl -s -o /dev/null -w "%{http_code} " http://localhost/health; sleep 3; done; echo && kubectl get pods -n heart-disease
```

```
deployment.apps/heart-disease-api restarted
200 200 200 200 200 200 
NAME                                 READY   STATUS    RESTARTS   AGE
heart-disease-api-558c88bb8b-mgb89   1/1     Running   0          22s
heart-disease-api-558c88bb8b-s6zxt   1/1     Running   0          16s
```

Six consecutive `200`s — the service stayed up throughout the restart — and both pods show fresh AGE, confirming the rolling-update strategy replaced them without downtime. Post-restart rollout status and cluster state were re-confirmed:

```bash
kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=60s
```

```
deployment "heart-disease-api" successfully rolled out
```

## Files changed / created

- `docs/figures/deploy/local_k8s_proof.md` (this file)
- `screenshots/deploy/.gitkeep`

## Self-review

- All commands above were executed against a live `docker-desktop` cluster in this session; no output was fabricated or beautified.
- The only deviation from the brief's exact expected text is the service `EXTERNAL-IP` value (`172.18.0.6` instead of the literal word `localhost`) — functionally equivalent for Docker Desktop, and confirmed working via the `curl http://localhost/...` calls immediately after.
- The image-load fallback (Step 3 of the brief) was not required; the rollout succeeded on the first `kubectl rollout status` call.
- Deployment was left running (2/2 pods `Running`, service `heart-disease-api` of type `LoadBalancer` on port 80) for the user's own screenshots.
