---

# 🚀 MLOps Pipeline MVP Checklist

Track your progress towards a Minimum Viable Product (MVP) for your end-to-end MLOps pipeline.  
Check each item as you complete the corresponding task!

---

## 1. Infrastructure & CI/CD
- [ ] Terraform configuration files for AWS resources (ECR, IAM) are created and committed
- [ ] AWS ECR repository and IAM role are provisioned via `terraform apply`
- [ ] GitHub Actions workflow builds and pushes Docker image to ECR
- [ ] Project documentation includes IaC and CI process instructions

## 2. Kubernetes Deployment
- [ ] Kubernetes manifests for Deployment, Service, and Ingress are created
- [ ] CI pipeline deploys FastAPI service to EKS after build
- [ ] FastAPI service is accessible externally
- [ ] Deployment instructions are included in documentation

## 3. Dependency Management
- [ ] All dependencies updated to latest compatible versions using Poetry
- [ ] All tests pass after updates
- [ ] `poetry.lock` file is updated and committed
- [ ] Dependency update instructions are documented

## 4. Feature Store Implementation
- [ ] Feature store schema is defined and documented
- [ ] Airflow ETL writes features to S3 (offline) and DynamoDB (online)
- [ ] Training pipeline reads from S3 feature store
- [ ] FastAPI service queries DynamoDB for predictions
- [ ] Feature store documentation is updated

## 5. Airflow Orchestration
- [ ] ETL DAG (EMR/Spark → S3) is created and runs successfully
- [ ] Model retraining DAG (SageMaker → MLflow) is created and runs successfully
- [ ] Retraining DAG updates MLflow Model Registry after training
- [ ] Pipeline DAGs and AWS integration are documented

## 6. MLflow Production Setup
- [ ] MLflow Tracking Server is deployed on EKS (with RDS and S3)
- [ ] S3 bucket and RDS instance provisioned via Terraform
- [ ] Training pipeline logs to remote MLflow server
- [ ] Models versioned and managed via MLflow Model Registry
- [ ] FastAPI service loads production model from registry
- [ ] MLflow architecture and config are documented

## 7. GitOps with Argo CD
- [ ] Argo CD Application manifest is created and applied
- [ ] CI pipeline updates Kubernetes manifest and commits changes
- [ ] Argo CD automatically triggers deployment to EKS
- [ ] GitOps flow is documented

## 8. Monitoring & Observability
- [ ] FastAPI service exposes `/metrics` endpoint
- [ ] Prometheus and Grafana stack deployed on EKS and scraping metrics
- [ ] Data drift detection is implemented and pushing metrics
- [ ] Grafana dashboard visualizes key metrics
- [ ] Alerts fire when data drift is detected
- [ ] Monitoring setup is documented

---