# Monitoring stack proof (2026-07-11)

Verbatim transcript of the v2 rollout and monitoring-stack verification on
the local Docker Desktop Kubernetes cluster (2 nodes, v1.36.1). Long build
output is trimmed to the final lines; everything shown is as observed.

The image-load fallback (docker save → ctr import into the nodes) was NOT
needed  -  the locally built `heart-disease-api:v2` was visible to the
cluster directly, and the rolling update succeeded on the first apply.

## 1. Build the instrumented v2 image

```console
$ docker build -t heart-disease-api:v2 -f infra/Dockerfile . 2>&1 | tail -5
#14 unpacking to docker.io/library/heart-disease-api:v2
#14 unpacking to docker.io/library/heart-disease-api:v2 3.0s done
#14 DONE 11.2s

View build details: docker-desktop://dashboard/build/desktop-linux/desktop-linux/yahh76mouxen1qtbe5ptd7feq

$ docker images heart-disease-api
IMAGE                      ID             DISK USAGE   CONTENT SIZE   EXTRA
heart-disease-api:latest   18069d8a0ef3        676MB          155MB
heart-disease-api:v1       b2fb0b9c3784        676MB          155MB
heart-disease-api:v2       172d0590e33a        677MB          155MB
```

## 2. Rolling update v1 → v2 (zero downtime)

```console
$ kubectl apply -f infra/k8s/deployment.yaml && kubectl rollout status deployment/heart-disease-api -n heart-disease --timeout=180s
deployment.apps/heart-disease-api configured
Waiting for deployment "heart-disease-api" rollout to finish: 1 out of 2 new replicas have been updated...
Waiting for deployment "heart-disease-api" rollout to finish: 1 out of 2 new replicas have been updated...
Waiting for deployment "heart-disease-api" rollout to finish: 1 out of 2 new replicas have been updated...
Waiting for deployment "heart-disease-api" rollout to finish: 1 old replicas are pending termination...
Waiting for deployment "heart-disease-api" rollout to finish: 1 old replicas are pending termination...
Waiting for deployment "heart-disease-api" rollout to finish: 1 old replicas are pending termination...
deployment "heart-disease-api" successfully rolled out
```

## 3. Live /metrics with the custom domain counter

```console
$ curl -s http://localhost/metrics | grep -E "^# HELP (http_requests_total|heart_disease_predictions_total)" && curl -s -X POST http://localhost/predict -H "Content-Type: application/json" -d @api/sample_request.json > /dev/null && curl -s http://localhost/metrics | grep heart_disease_predictions_total
# HELP heart_disease_predictions_total Predictions served, by predicted risk label
# HELP http_requests_total Total number of requests by method, status and handler.
# HELP heart_disease_predictions_total Predictions served, by predicted risk label
# TYPE heart_disease_predictions_total counter
heart_disease_predictions_total{risk_label="no disease"} 1.0
```

## 4. Deploy Prometheus + Grafana (fully provisioned, one apply)

```console
$ kubectl apply -f infra/k8s/monitoring/ && kubectl rollout status deployment/prometheus -n heart-disease --timeout=180s && kubectl rollout status deployment/grafana -n heart-disease --timeout=180s && kubectl get pods,svc -n heart-disease
configmap/grafana-datasources created
configmap/grafana-dashboard-provider created
configmap/grafana-dashboards created
deployment.apps/grafana created
service/grafana created
configmap/prometheus-config created
deployment.apps/prometheus created
service/prometheus created
Waiting for deployment "prometheus" rollout to finish: 0 of 1 updated replicas are available...
deployment "prometheus" successfully rolled out
deployment "grafana" successfully rolled out
NAME                                   READY   STATUS    RESTARTS   AGE
pod/grafana-6644dfc7dd-cc7rg           1/1     Running   0          52s
pod/heart-disease-api-7b8bdb58-pz6cv   1/1     Running   0          99s
pod/heart-disease-api-7b8bdb58-qkxkk   1/1     Running   0          86s
pod/prometheus-647889bd95-2dwv2        1/1     Running   0          52s

NAME                        TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
service/grafana             LoadBalancer   10.96.217.107   172.18.0.8    3000:30594/TCP   52s
service/heart-disease-api   LoadBalancer   10.96.75.197    172.18.0.6    80:32503/TCP     3h51m
service/prometheus          LoadBalancer   10.96.128.199   172.18.0.7    9090:31148/TCP   52s
```

## 5. Traffic generation

```console
$ for i in $(seq 1 30); do curl -s -X POST http://localhost/predict -H "Content-Type: application/json" -d @api/sample_request.json > /dev/null; curl -s http://localhost/health > /dev/null; done; echo done
done
```

## 6. Prometheus scrape target is up

```console
$ curl -s "http://localhost:9090/api/v1/targets" | grep -o '"health":"[a-z]*"'
"health":"up"
```

## 7. PromQL queries return live data

```console
$ curl -s "http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total%5B1m%5D))" | head -c 300
{"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[1783758766.352,"0.8470644983805677"]}]}}

$ curl -s "http://localhost:9090/api/v1/query?query=heart_disease_predictions_total" | head -c 300
{"status":"success","data":{"resultType":"vector","result":[{"metric":{"__name__":"heart_disease_predictions_total","instance":"heart-disease-api.heart-disease.svc:80","job":"heart-disease-api","risk_label":"no disease"},"value":[1783758766.876,"10"]}]}}
```

## 8. Grafana healthy with the provisioned dashboard

```console
$ curl -s http://localhost:3000/api/health && echo && curl -s "http://localhost:3000/api/dashboards/uid/heart-disease-api" | head -c 200
{
  "database": "ok",
  "version": "12.3.0",
  "commit": "20051fb1fc604fc54aae76356da1c14612af41d0"
}
{"meta":{"type":"db","canSave":false,"canEdit":false,"canAdmin":false,"canStar":false,"canDelete":false,"slug":"heart-disease-api","url":"/d/heart-disease-api/heart-disease-api","expires":"0001-01-01T

$ curl -s "http://localhost:3000/api/dashboards/uid/heart-disease-api" | grep -o '"title":"Heart Disease API"'
"title":"Heart Disease API"
```
