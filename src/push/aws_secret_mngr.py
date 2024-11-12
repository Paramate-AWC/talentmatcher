# If you store your credentials in AWS Secrets Manager
import json
import boto3

def get_rds_credentials_from_secrets_manager(secret_name, region_name):
    session = boto3.Session(region_name=region_name)
    client = session.client('secretsmanager')
    secret_value = client.get_secret_value(SecretId=secret_name)
    return json.loads(secret_value['SecretString'])
