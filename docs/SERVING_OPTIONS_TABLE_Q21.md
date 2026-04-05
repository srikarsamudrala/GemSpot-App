## GemSpot Serving Options Table

Measured on Chameleon Cloud.

| Option | Endpoint URL | Model version | Code version | Hardware | p50/p95 latency | Throughput | Error rate | Concurrency tested | Compute instance type | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_http` | `http://129.114.27.244:8050/recommend` | `mock-v1.0` | `182836a9` | CPU | `4.91 / 9.92 ms` | `38.64 req/s` | `0` | `20 users, ramp 5, 30s` | `m1.medium` | Best throughput, simplest deployment |
| `xgboost_workers2` | `http://129.114.27.244:8051/recommend` | `mock-v1.0` | `182836a9` | CPU | `5.10 / 9.70 ms` | `15.42 req/s` | `0` | `20 users, ramp 5, 30s` | `m1.medium` | Slightly better tail latency than baseline, but much lower throughput |
| `onnx_cpu` | `http://129.114.27.244:8052/recommend` | `mock-v1.0` | `182836a9` | CPU | `4.02 / 4.19 ms` | `15.12 req/s` | `0` | `20 users, ramp 5, 30s` | `m1.medium` | Best latency; strongest model-level optimization |
| `onnx_workers4` | `http://129.114.27.244:8053/recommend` | `mock-v1.0` | `182836a9` | CPU | `4.07 / 4.35 ms` | `15.77 req/s` | `0` | `20 users, ramp 5, 30s` | `m1.medium` | Combined optimization; slightly higher throughput than ONNX 1-worker |
| `infra_rightsized` | `http://129.114.27.109:8052/recommend` | `mock-v1.0` | `182836a9` | CPU | `4.29 / 4.45 ms` | `15.67 req/s` | `0` | `20 users, ramp 5, 30s` | `m1.large` | Larger VM did not materially improve performance; smaller CPU flavor appears sufficient |
