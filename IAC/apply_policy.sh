aws iam put-user-policy \
  --user-name terraform_erwan \
  --policy-name TerraformEC2Policy \
  --policy-document file://policy/patch-policy.json

