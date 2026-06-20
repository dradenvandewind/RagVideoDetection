terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ──────────────────────────────────────────────
# Data sources
# ──────────────────────────────────────────────

# Latest Ubuntu 24.04 ARM64 (Graviton2 compatible)
data "aws_ami" "ubuntu_arm64" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
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
    description = "Ollama (optional direct access)"
    from_port   = 11434
    to_port     = 11434
    protocol    = "tcp"
    cidr_blocks = [var.allowed_api_cidr]
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
# EBS Volume (data persistence - ChromaDB, models)
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
# EC2 Instance — t4g.small (Graviton2 ARM64)
# ──────────────────────────────────────────────

resource "aws_instance" "rag_server" {
  ami                    = data.aws_ami.ubuntu_arm64.id
  instance_type          = "t4g.small"
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.rag_api.id]
  key_name               = aws_key_pair.deployer.key_name

  root_block_device {
    volume_size           = 45
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  # User data minimal — Ansible s'occupe du reste
  user_data = <<-EOF
    #!/bin/bash
    apt-get update -y
    apt-get install -y python3 python3-pip
  EOF

  tags = {
    Name        = "${var.project_name}-server"
    Environment = var.environment
    Project     = var.project_name
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
  instance = aws_instance.rag_server.id
  domain   = "vpc"
  tags     = { Name = "${var.project_name}-eip" }
}

# ──────────────────────────────────────────────
# Ansible inventory (generated automatically)
# ──────────────────────────────────────────────

resource "local_file" "ansible_inventory" {
  content = templatefile("${path.module}/inventory.tpl", {
    public_ip        = aws_eip.rag_server.public_ip
    ssh_user         = "ubuntu"
    ssh_private_key  = var.ssh_private_key_path
    project_name     = var.project_name
    api_port         = var.api_port
  })
  filename        = "${path.module}/../ansible/inventory.ini"
  file_permission = "0644"
}
