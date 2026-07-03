# AWS Deployment Guide

Deploy the Credit Risk API to AWS ECS Fargate — serverless containers, no EC2 to manage.

## Architecture

```
GitHub Actions ─► ECR (image registry) ─► ECS Fargate (API container)
                                              │
                                         ALB (load balancer)
                                              │
                                         Public HTTPS endpoint
```

## Prerequisites

- AWS account with CLI configured (`aws configure`)
- Docker installed locally
- An ECR repository and ECS cluster (created below)

## One-Time Setup

### 1. Create the ECR repository

```bash
aws ecr create-repository --repository-name credit-risk-platform --region us-east-1
```

### 2. Build and push the image

```bash
# Authenticate Docker with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build (make sure the model artifacts exist first: python scripts/run_pipeline.py)
docker build -t credit-risk-platform .

# Tag and push
docker tag credit-risk-platform:latest \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-platform:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-platform:latest
```

### 3. Create the ECS cluster

```bash
aws ecs create-cluster --cluster-name credit-risk-cluster
```

### 4. Register the task definition

Edit `ecs-task-definition.json` — replace `<ACCOUNT_ID>` and `<REGION>` — then:

```bash
aws ecs register-task-definition --cli-input-json file://deploy/aws/ecs-task-definition.json
```

### 5. Create the service

```bash
aws ecs create-service \
  --cluster credit-risk-cluster \
  --service-name credit-risk-api \
  --task-definition credit-risk-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<SUBNET_ID>],securityGroups=[<SG_ID>],assignPublicIp=ENABLED}"
```

The security group must allow inbound TCP 8000 (or put an ALB in front for HTTPS on 443).

## Continuous Deployment

`.github/workflows/deploy.yml` deploys on manual trigger (Actions tab → Deploy to AWS → Run workflow).

Required repository secrets (Settings → Secrets and variables → Actions):

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ACCOUNT_ID` | 12-digit account ID |

The IAM user needs: `AmazonEC2ContainerRegistryPowerUser` and `AmazonECS_FullAccess` (scope down for production).

## Cost Notes

- Fargate 0.5 vCPU / 1GB: ~$18/month if running 24/7
- To demo cheaply: `aws ecs update-service --desired-count 0` when not in use, `--desired-count 1` to bring it back
- ECR storage: pennies for one image

## Verify Deployment

```bash
curl http://<PUBLIC_IP>:8000/health
curl http://<PUBLIC_IP>:8000/docs      # interactive API docs
curl http://<PUBLIC_IP>:8000/metrics   # Prometheus metrics
```
