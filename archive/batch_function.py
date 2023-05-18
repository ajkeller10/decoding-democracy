import boto3
import json

# Create a batch client
batch = boto3.client('batch')

# Define the input and output S3 bucket and file paths
input_bucket = 'decoding-democracy'
input_key = 'unembedded.csv'
output_bucket = 'decoding-democracy'
output_key = 'embedded.csv'

# Define the event payload
event = {
    'input_bucket': input_bucket,
    'input_key': input_key,
    'output_bucket': output_bucket,
    'output_key': output_key
}

# Define the job definition
job_definition = 'embedding sentences for df with list of sentences saved as sentence'

# Define the job name
job_name = 'embed_decoding_democracy'

# Define the job queue
job_queue = 'embed_decoding_democracy_queue'

# Submit the job
response = batch.submit_job(
    jobName=job_name,
    jobQueue=job_queue,
    jobDefinition=job_definition,
    parameters=event
)

# Print the job ID
print('Submitted batch job:', response['jobId'])
