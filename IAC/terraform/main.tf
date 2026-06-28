terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    # ✅ ADD: local provider needed for local_file
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  skip_requesting_account_id  = false
  skip_credentials_validation = false
  skip_metadata_api_check     = false

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
    }
  }
}

# ──────────────────────────────────────────────
# Data sources
# ──────────────────────────────────────────────

# ✅ SSM Parameter Store — AWS official method to locate the ARM64 DLAMI
#    Works in all regions without hardcoding a name or AMI ID.
#    Contains: NVIDIA driver + CUDA 12 + nvidia-container-toolkit + Docker
#    Compatible: g5g (Graviton2 + T4G GPU)
data "aws_ssm_parameter" "dlami_arm64" {
  name = "/aws/service/deeplearning/ami/arm64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
}

# ──────────────────────────────────────────────
# VPC & Networking
# ──────────────────────────────────────────────

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${var.project_name}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = { Name = "${var.project_name}-subnet-public" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${var.project_name}-rt-public" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# ──────────────────────────────────────────────
# Security Group
# ──────────────────────────────────────────────

resource "aws_security_group" "rag_api" {
  name        = "${var.project_name}-sg"
  description = "Security group for RAG LlamaIndex API"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  ingress {
    description = "RAG API"
    from_port   = var.api_port
    to_port     = var.api_port
    protocol    = "tcp"
    cidr_blocks = [var.allowed_api_cidr]
  }

  ingress {
    description = "Ollama"
    from_port   = 11434
    to_port     = 11434
    protocol    = "tcp"
    cidr_blocks = [var.allowed_api_cidr]
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP Alt"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-sg" }
}

# ──────────────────────────────────────────────
# SSH Key Pair
# ──────────────────────────────────────────────

resource "aws_key_pair" "deployer" {
  key_name   = "${var.project_name}-key"
  public_key = file(var.ssh_public_key_path)
}

# ──────────────────────────────────────────────
# EBS Volume (ChromaDB + HuggingFace models)
# ──────────────────────────────────────────────

resource "aws_ebs_volume" "data" {
  availability_zone = "${var.aws_region}a"
  size              = var.data_volume_size_gb
  type              = "gp3"
  throughput        = 125
  iops              = 3000
  encrypted         = true

  tags = { Name = "${var.project_name}-data" }
}

# ──────────────────────────────────────────────
# User Data — bootstrap minimal (Docker + signal)
# Ansible handles NVIDIA after boot
# ──────────────────────────────────────────────

locals {
  user_data = <<-EOF
    #!/usr/bin/env bash
    set -eux

    # System update
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y ca-certificates curl gnupg lsb-release

    # Docker
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
      > /etc/apt/sources.list.d/docker.list

    apt-get update -y
    apt-get install -y \
      docker-ce docker-ce-cli containerd.io \
      docker-buildx-plugin docker-compose-plugin

    systemctl enable --now docker
    usermod -aG docker ubuntu

    # ✅ Signal for Ansible: wait for this file before continuing
    touch /tmp/bootstrap_done
  EOF
}

# ──────────────────────────────────────────────
# EC2 Instance — Spot g5g (ARM64 + GPU NVIDIA T4G)
# ──────────────────────────────────────────────

resource "aws_instance" "rag_server" {
  ami                    = data.aws_ssm_parameter.dlami_arm64.value
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.rag_api.id]
  key_name               = aws_key_pair.deployer.key_name

  
  user_data = local.user_data

  
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price                      = var.spot_max_price
      spot_instance_type             = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }

  root_block_device {
    volume_size           = 120
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = {
    Name        = "${var.project_name}-server"
    Environment = var.environment
  }
}

resource "aws_volume_attachment" "data" {
  device_name  = "/dev/sdf"
  volume_id    = aws_ebs_volume.data.id
  instance_id  = aws_instance.rag_server.id
  force_detach = false
}

# ──────────────────────────────────────────────
# Elastic IP
# ──────────────────────────────────────────────

resource "aws_eip" "rag_server" {
  # ✅ NOTE: do not set instance= here because the remote-exec provisioner
  #    in aws_instance already references aws_eip → Terraform handles the dependency
  domain = "vpc"
  tags   = { Name = "${var.project_name}-eip" }
}

resource "aws_eip_association" "rag_server" {
  instance_id   = aws_instance.rag_server.id
  allocation_id = aws_eip.rag_server.id
}

# ──────────────────────────────────────────────
# Ansible inventory (generated automatically)
# ──────────────────────────────────────────────

resource "local_file" "ansible_inventory" {
  content = templatefile("${path.module}/inventory.tpl", {
    public_ip       = aws_eip.rag_server.public_ip
    ssh_user        = "ubuntu"
    ssh_private_key = var.ssh_private_key_path
    project_name    = var.project_name
    api_port        = var.api_port
  })
  filename        = "${path.module}/../ansible/inventory.ini"
  file_permission = "0644"
}
# ──────────────────────────────────────────────
# null_resource — provisioners AFTER association
# EIP to avoid aws_instance ↔ aws_eip cycle
# ──────────────────────────────────────────────

resource "null_resource" "ansible_provision" {
  triggers = {
    instance_id = aws_instance.rag_server.id
    public_ip   = aws_eip.rag_server.public_ip
  }

  depends_on = [
    aws_eip_association.rag_server,
    aws_volume_attachment.data,
  ]

  provisioner "remote-exec" {
    inline = [
      "echo 'Waiting for bootstrap...'",
      "until [ -f /tmp/bootstrap_done ]; do sleep 5; done",
      "echo 'Bootstrap done.'"
    ]

    connection {
      type        = "ssh"
      user        = "ubuntu"
      private_key = file(var.ssh_private_key_path)
      host        = aws_eip.rag_server.public_ip
      timeout     = "10m"
    }
  }

  provisioner "local-exec" {
    command = <<-EOT
      ansible-playbook \
        -i '${aws_eip.rag_server.public_ip},' \
        --private-key ${var.ssh_private_key_path} \
        -u ubuntu \
        --ssh-extra-args='-o StrictHostKeyChecking=no' \
        ../ansible/playbooks/setup_gpu.yml
    EOT
  }
}