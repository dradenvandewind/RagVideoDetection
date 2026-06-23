variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-3" # Paris
}
variable "spot_max_price" {
  description = "Prix maximum Spot en USD/heure"
  type        = string
  default     = "0.012"
}


variable "project_name" {
  description = "Project name (used in resource names and tags)"
  type        = string
  default     = "rag-llamaindex"
}

variable "environment" {
  description = "Environment (prod, staging, dev)"
  type        = string
  default     = "prod"
}

variable "ssh_public_key_path" {
  description = "Path to your SSH public key"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "ssh_private_key_path" {
  description = "Path to your SSH private key (used in Ansible inventory)"
  type        = string
  default     = "~/.ssh/id_rsa"
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH into the instance (restrict to your IP)"
  type        = string
  default     = "0.0.0.0/0" # À restreindre à votre IP : ex. "1.2.3.4/32"
}

variable "allowed_api_cidr" {
  description = "CIDR allowed to reach the API"
  type        = string
  default     = "0.0.0.0/0"
}

variable "api_port" {
  description = "Port exposed by the FastAPI app"
  type        = number
  default     = 8000
}

variable "data_volume_size_gb" {
  description = "EBS data volume size in GB (ChromaDB + HuggingFace models cache)"
  type        = number
  default     = 50
}
variable "instance_type" {
  description = "EC2 instance type (ARM64 Graviton)"
  type        = string
  default     = "t4g.medium"
}