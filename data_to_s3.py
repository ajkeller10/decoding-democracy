import boto3

#aws s3api create-bucket --bucket your-bucket-name --region your-region
#aws s3api delete-bucket --bucket your-bucket-name

#creating fargate instance under aws batch
#maximum vcpu at 4

# Set your AWS credentials
aws_access_key_id = 'your_access_key_id'
aws_secret_access_key = 'your_secret_access_key'

# Set the S3 bucket name and key for your CSV file
bucket_name = 'decoding_democracy_bucket'
file_key = 'unembedded.csv'

# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Upload the CSV file to S3
s3.upload_file('data_for_s3.pickle', 'decoding-democracy', 'unembedded.csv')

#downloading
file_key = 'embedded.csv'

# Download the CSV file from S3
s3.download_file(bucket_name, file_key, 'your-local-file.csv')