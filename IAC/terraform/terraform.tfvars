# Copier ce fichier en terraform.tfvars et adapter les valeurs

aws_region           = "eu-west-3"          # Paris (Graviton disponible)
project_name         = "rag-llamaindex"
environment          = "prod"

ssh_public_key_path  = "~/.ssh/id_rsa.pub"
ssh_private_key_path = "~/.ssh/id_rsa"

# ⚠️  Restreindre à votre IP pour la sécurité : "x.x.x.x/32"
allowed_ssh_cidr     = "0.0.0.0/0"
allowed_api_cidr     = "0.0.0.0/0"

api_port             = 8000
data_volume_size_gb  = 50                   # ChromaDB + HF models cache
instance_type = "t4g.small"
