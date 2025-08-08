# How to Use This Terraform Code

This guide explains how to set up and manage your MLOps infrastructure using the provided `main.tf` file.

---

## 1. Save the Code

Save the Terraform configuration content as a file named **`main.tf`** in a new, empty directory.

---

## 2. Environment Variables

Your AWS credentials should be stored in an `.env` file. To make them accessible to Terraform, you need to load them into your shell's environment.

Create a file named `.env` with the following content:

```bash
export AWS_ACCESS_KEY_ID="<your_access_key>"
export AWS_SECRET_ACCESS_KEY="<your_secret_key>"
```

Then, run the following command in your terminal before using Terraform:

```bash
source .env
```

---

## 3. Variable File

Create a file named **`terraform.tfvars`** to customize any variables defined in the `main.tf` file. For example:

```
project_name = "my-awesome-mlops-project"
aws_region = "us-west-2"
```

---

## 4. Run Terraform

Navigate to the directory containing your Terraform files and execute the following commands in order.

### Initialize Terraform

This command downloads the necessary provider plugins.

```bash
terraform init
```

### Plan Changes

This command shows you what resources Terraform will create, modify, or destroy.

```bash
terraform plan
```

### Apply Changes

This command applies the plan and builds the infrastructure on AWS. The `--auto-approve` flag bypasses the confirmation prompt.

```bash
terraform apply --auto-approve
```

---

## 5. Clean Up

To destroy all the resources provisioned by this Terraform configuration, run the following command:

```bash
terraform destroy --auto-approve
