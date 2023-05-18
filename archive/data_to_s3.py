import boto3

# Set your AWS credentials
aws_access_key_id = 'your_access_key_id'
aws_secret_access_key = 'your_secret_access_key'

# Set the S3 bucket name and key for your CSV file
bucket_name = 'a7-olpinney'
file_key = 'ami'

# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# Upload the CSV file to S3
s3.upload_file('ami_pickle.pickle', bucket_name, file_key)

#downloading

# Download the CSV file from S3
s3.download_file(bucket_name, file_key, 'your-local-file.csv')