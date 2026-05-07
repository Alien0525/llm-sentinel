# LLM Sentinel — Results Summary
_Generated: 2026-05-06 20:17_

## baseline_attack_log_20260429_153910_corrected.json
| Metric | Value |
|--------|-------|
| Total requests | 21 |
| BLOCKED | 0 (0%) |
| COMPLIED | 18 |
| Avg latency | 0ms |

## week5_benchmark_20260506_235906.json
| Metric | Value |
|--------|-------|
| Total requests | 42 |
| BLOCKED | 38 (90%) |
| COMPLIED | 4 |
| Base64 detected | 21 |
| Layer1 blocks | 30 |
| Layer3 blocks | 8 |
| Avg latency | 4671ms |

## week5_benchmark_20260429_164122.json
| Metric | Value |
|--------|-------|
| Total requests | 42 |
| BLOCKED | 28 (67%) |
| COMPLIED | 14 |
| Base64 detected | 21 |
| Layer2 blocks | 28 |
| Avg latency | 2829ms |

## Before vs After Comparison
| Metric | Baseline (unprotected) | Protected (all 3 layers) |
|--------|------------------------|--------------------------|
| Block rate (plain) | 0% | 86% |
| Block rate (base64 encoded) | 0% | 95% |
| Base64 encodings detected | N/A | 21/21 |
| Layer 1 blocks | N/A | 30 |
| Layer 2 blocks | N/A | 0* |
| Layer 3 blocks | N/A | 8 |
| Avg blocked latency | N/A | ~300ms |
| Avg complied latency | 5-10s | ~13s |
| p50 latency (all requests) | N/A | 297ms |
| p95 latency (all requests) | N/A | 17564ms |

_* Layer 2 not triggered when Layer 1 catches the request first_
