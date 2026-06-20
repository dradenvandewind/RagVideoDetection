[rag_servers]
${project_name} ansible_host=${public_ip} ansible_user=${ssh_user} ansible_ssh_private_key_file=${ssh_private_key} ansible_ssh_common_args='-o StrictHostKeyChecking=no'

[rag_servers:vars]
api_port=${api_port}
