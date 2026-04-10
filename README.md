# LLM Sentinel

Cloud-native defense framework against adversarial attacks on Large Language Models.

---

## Overview

LLM Sentinel simulates adversarial attacks on a cloud-hosted LLM and measures the effectiveness of a multi-layered defense architecture. The system intercepts malicious prompts before they reach the target model and validates outputs before returning them to the user.

The project benchmarks security in two scenarios:

- **Baseline** — direct attacks on the unprotected LLM to establish vulnerability metrics
- **Protected** — same attacks routed through all three defense layers to measure improvement

---

## Team

| Name | Role | Focus |
|------|------|-------|
| Varsha | Security Architecture Lead | Three-layer defense design, AWS infrastructure, ML pipeline |
| Krisha Joshi | ML and Data Lead | Dataset curation, model training, TF-IDF classifier |
| Aman NS | Cloud and Integration Lead | AWS setup, API Gateway, Lambda, DynamoDB, CloudWatch |
| Sisvanth | Backend and Testing Lead | Attack simulation, baseline testing, logging pipeline |

---

## Architecture

```
User Request
     |
     v
API Gateway (POST /prompt)
     |
     v
Layer 1: Lambda + Custom ML Classifier
     |  Blocks known patterns, Base64/hex encoded payloads
     v
Layer 2: Amazon Bedrock Guardrails (Input)
     |  Handles sophisticated jailbreaks and semantic attacks
     v
Target LLM — EC2 (t2.micro) + Ollama + TinyLlama
     |
     v
Layer 3: Amazon Bedrock Guardrails (Output)
     |  Scans response for PII, system prompt leakage, secrets
     v
Response + Logging
     |
     +-- DynamoDB (attack logs and metrics)
     +-- S3 (bypass payloads for retraining)
     +-- CloudWatch + SNS (monitoring and alerts)
     |
     v
Streamlit Dashboard (EC2)
```

---

## Defense Layers

### Layer 1 — Edge Filter

A custom binary classifier trained on labeled prompt injection datasets. Deployed as a containerized Lambda function via Amazon ECR. Provides fast, low-latency blocking for known attack patterns including Base64 and hex-encoded payloads.

- Model: Scikit-learn TF-IDF vectorizer + Logistic Regression
- Trained locally, saved as `model.pkl`, containerized with Docker
- Deployed to AWS Lambda for real-time inference

### Layer 2 — Semantic Guard

Amazon Bedrock Guardrails applied to the incoming prompt after it passes Layer 1. Detects sophisticated jailbreaks and injection attempts that pattern-based classifiers miss by analyzing semantic meaning.

### Layer 3 — Egress Auditor

A second Bedrock Guardrails configuration applied to the LLM output before it is returned to the user. Prevents accidental leakage of system prompts, secrets, or PII in responses.

---

## AWS Services

| Component | Service | Purpose |
|-----------|---------|---------|
| LLM Hosting | EC2 t2.micro | Runs Ollama with TinyLlama |
| API Layer | API Gateway (HTTP API) | Public endpoint with rate limiting |
| Edge Filter | Lambda + ECR | Runs ML classifier container |
| Semantic Guard | Bedrock Guardrails | Input protection |
| Egress Auditor | Bedrock Guardrails | Output protection |
| Storage | DynamoDB | Attack logs and metrics |
| Bypass Storage | S3 | Attack payloads for retraining |
| Container Registry | ECR | Stores ML model Docker image |
| Dashboard | EC2 (same instance) | Streamlit UI |
| Monitoring | CloudWatch | Logs and real-time metrics |
| Alerting | CloudWatch Alarms + SNS | Email alerts on high attack rates |
| Retraining Trigger | S3 Event Notifications | Triggers retraining pipeline |

---

## ML Pipeline

1. Collect and label prompt injection datasets from HuggingFace
2. Train locally: `sklearn.pipeline` with `TfidfVectorizer` and `LogisticRegression`
3. Evaluate accuracy — target greater than 85%
4. Save trained model as `model.pkl`
5. Build Docker container image
6. Push image to Amazon ECR
7. Deploy via AWS Lambda for real-time prompt classification

---

## Setup

### Prerequisites

- AWS account with free tier active
- Python 3.10+
- Docker
- AWS CLI configured locally

### EC2 and Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama
sudo systemctl start ollama

# Pull TinyLlama
ollama pull tinyllama

# Expose Ollama on all interfaces (add to systemd override)
sudo systemctl edit ollama
# Add:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0:11434"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

If the EC2 instance has insufficient RAM (t2.micro has 1GB), add swap:

```bash
sudo swapoff /swapfile 2>/dev/null; sudo rm -f /swapfile
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Lambda Proxy

The Lambda function forwards prompts from API Gateway to Ollama on EC2 and returns the response. Set the environment variable `OLLAMA_EC2_URL` to `http://<EC2_IP>:11434`.

### Test the endpoint

```bash
curl -X POST \
  https://<API_ID>.execute-api.us-east-1.amazonaws.com/dev/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?"}'
```

---

## Security Notes

- Port 11434 (Ollama) must be restricted to trusted IPs only — do not expose it publicly without authentication
- Assign an Elastic IP to the EC2 instance so the Ollama URL does not change on restarts
- Bedrock Guardrails is pay-per-use — batch test prompts during development rather than testing continuously
- Set a throttle rate on the API Gateway stage to prevent runaway costs

---

## Budget Estimate

| Service | Monthly Cost |
|---------|-------------|
| EC2 t2.micro | $0 (free tier) |
| Lambda | $0 (free tier, 1M requests) |
| API Gateway | $0 (free tier, 1M requests) |
| DynamoDB | $0 (free tier, 25GB) |
| S3 | $0 (free tier, 5GB) |
| ECR | $0 (free tier, 500MB) |
| Bedrock Guardrails | $2 - $5 (pay-per-use) |
| CloudWatch | $0 (free tier) |
| SNS | $0 (free tier, 1,000 emails) |
| **Total** | **$2 - $17** |

---

## References

- [Amazon Bedrock Guardrails documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
- [Ollama](https://ollama.com)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [HuggingFace prompt injection datasets](https://huggingface.co/datasets?search=prompt+injection)
- [Scikit-learn TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)