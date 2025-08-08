# main.tf

# ------------------------------------------------------------------------------
# Provider Configuration
# ------------------------------------------------------------------------------
# The AWS provider is configured to use environment variables for credentials.
# The user must set these variables (e.g., in a .env file) before running terraform.
# e.g., export AWS_ACCESS_KEY_ID="..." and export AWS_SECRET_ACCESS_KEY="..."
provider "aws" {
  region = var.aws_region
}

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------
# Define all configurable values here to make the code reusable.
variable "aws_region" {
  description = "The AWS region to deploy resources in."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "A unique name for the project, used to name resources."
  type        = string
  default     = "mlops-pipeline"
}

variable "mwaa_version" {
  description = "The MWAA version to use."
  type        = string
  default     = "2.8.1"
}

# ------------------------------------------------------------------------------
# VPC and Networking Setup
# A new VPC is created to isolate the MLOps infrastructure.
# ------------------------------------------------------------------------------
resource "aws_vpc" "mlops_vpc" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_subnet" "private_subnet" {
  count             = 2
  vpc_id            = aws_vpc.mlops_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = "${var.aws_region}${count.index == 0 ? "a" : "b"}"
  tags = {
    Name = "${var.project_name}-private-subnet-${count.index + 1}"
  }
}

resource "aws_internet_gateway" "mlops_igw" {
  vpc_id = aws_vpc.mlops_vpc.id
  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_route_table" "mlops_rt" {
  vpc_id = aws_vpc.mlops_vpc.id
  tags = {
    Name = "${var.project_name}-rt"
  }
}

resource "aws_route" "default_route" {
  route_table_id         = aws_route_table.mlops_rt.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.mlops_igw.id
}

resource "aws_route_table_association" "private_rt_assoc" {
  count          = length(aws_subnet.private_subnet)
  subnet_id      = aws_subnet.private_subnet[count.index].id
  route_table_id = aws_route_table.mlops_rt.id
}

# ------------------------------------------------------------------------------
# IAM Roles & Policies (Security Core)
# ------------------------------------------------------------------------------
# MWAA Execution Role
resource "aws_iam_role" "mwaa_exec_role" {
  name = "${var.project_name}-mwaa-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "airflow.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "mwaa_s3_policy" {
  role       = aws_iam_role.mwaa_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess" # Or more restrictive
}

# EKS Cluster Role
resource "aws_iam_role" "eks_cluster_role" {
  name = "${var.project_name}-eks-cluster-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

# ------------------------------------------------------------------------------
# S3 Buckets (Data Lake & Storage Hub)
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "mwaa_dags_bucket" {
  bucket = "${var.project_name}-mwaa-dags-bucket"
  tags = {
    Name = "MWAA DAGs"
  }
}

resource "aws_s3_bucket_acl" "mwaa_dags_bucket_acl" {
  bucket = aws_s3_bucket.mwaa_dags_bucket.id
  acl    = "private"
}

resource "aws_s3_bucket" "mlops_data_lake" {
  bucket = "${var.project_name}-data-lake"
  tags = {
    Name = "Data Lake"
  }
}

resource "aws_s3_bucket" "mlops_artifacts" {
  bucket = "${var.project_name}-artifacts"
  tags = {
    Name = "Model and MLflow Artifacts"
  }
}

# ------------------------------------------------------------------------------
# Amazon ECR (Container Registry)
# ------------------------------------------------------------------------------
resource "aws_ecr_repository" "etl_repo" {
  name                 = "${var.project_name}-etl-repo"
  image_tag_mutability = "MUTABLE"
  tags = {
    Name = "ETL Image Repository"
  }
}

resource "aws_ecr_repository" "mlflow_repo" {
  name                 = "${var.project_name}-mlflow-repo"
  image_tag_mutability = "MUTABLE"
  tags = {
    Name = "MLflow Image Repository"
  }
}

# ------------------------------------------------------------------------------
# Amazon MWAA (Managed Apache Airflow)
# ------------------------------------------------------------------------------
resource "aws_mwaa_environment" "mlops_mwaa" {
  name                       = "${var.project_name}-mwaa"
  dag_s3_path                = "dags" # Assumes a 'dags' folder in the S3 bucket
  execution_role_arn         = aws_iam_role.mwaa_exec_role.arn
  environment_class          = "mw1.small"
  source_bucket_arn          = aws_s3_bucket.mwaa_dags_bucket.arn
  webserver_access_mode      = "PUBLIC_ONLY" # Change to PRIVATE_ONLY for production
  
  logging_configuration {
    dag_processing_logs {
      enabled = true
      log_level = "INFO"
    }
  }

  network_configuration {
    security_group_ids = [] # Add a security group ID
    subnet_ids         = aws_subnet.private_subnet[*].id
  }
}

# ------------------------------------------------------------------------------
# Amazon RDS (MLflow Backend Database)
# ------------------------------------------------------------------------------
resource "aws_db_instance" "mlflow_db" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "15.4"
  instance_class       = "db.t3.micro"
  db_name              = "mlflowdb"
  username             = "mlflowuser"
  password             = "mlflowpassword" # Use AWS Secrets Manager in production
  skip_final_snapshot  = true
}

# ------------------------------------------------------------------------------
# Amazon EMR (Spark Cluster for ETL)
# ------------------------------------------------------------------------------
resource "aws_emr_cluster" "mlops_emr" {
  name          = "${var.project_name}-emr-cluster"
  release_label = "emr-6.10.0"
  applications  = ["Spark"]
  
  ec2_attributes {
    subnet_id = aws_subnet.private_subnet[0].id
  }

  master_instance_group {
    instance_type  = "m5.xlarge"
    instance_count = 1
  }

  core_instance_group {
    instance_type  = "m5.xlarge"
    instance_count = 2
  }

  ebs_root_volume_size = 10
  
  service_role = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceRole" # Use a custom role in production
  # Job flow role needs to be specified for custom roles
  # job_flow_role = ...
}

# ------------------------------------------------------------------------------
# Amazon EKS (Kubernetes Cluster for Argo CD & MLflow)
# ------------------------------------------------------------------------------
resource "aws_eks_cluster" "mlops_eks" {
  name     = "${var.project_name}-eks-cluster"
  role_arn = aws_iam_role.eks_cluster_role.arn
  vpc_config {
    subnet_ids = aws_subnet.private_subnet[*].id
  }
}

resource "aws_eks_node_group" "mlops_eks_node_group" {
  cluster_name    = aws_eks_cluster.mlops_eks.name
  node_group_name = "${var.project_name}-node-group"
  subnet_ids      = aws_subnet.private_subnet[*].id
  instance_types  = ["t3.medium"]

  scaling_config {
    desired_size = 2
    max_size     = 3
    min_size     = 1
  }

  # This is a placeholder for a dedicated IAM role for the worker nodes
  node_role_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

# ------------------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------------------
output "mwaa_endpoint" {
  description = "The URL for the MWAA Airflow UI."
  value       = aws_mwaa_environment.mlops_mwaa.webserver_url
}

output "eks_cluster_name" {
  description = "The name of the EKS cluster."
  value       = aws_eks_cluster.mlops_eks.name
}

output "mlflow_db_endpoint" {
  description = "The endpoint of the MLflow PostgreSQL database."
  value       = aws_db_instance.mlflow_db.address
}
