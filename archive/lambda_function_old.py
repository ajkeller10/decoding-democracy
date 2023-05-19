import json
import boto3

# Initialize s3
s3_resource = boto3.resource('s3')

# Initialize DynamoDB resource via boto3
dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    '''
    lambda handler built for SQS event object
    '''
    
    source = event['Records'][0]['eventSourceARN']
    event = json.loads(event['Records'][0]['body'])

    # check results of survey
    if (event['time_elapsed']>3) and (len(event['freetext'])>0):

        #connect to s3
        bucket = 'hw5-olpinney'
        bucket_resource = s3_resource.Bucket(bucket)

        #send to s3
        key_id=event['user_id']+event['timestamp']+".json"
        bucket_resource.put_object(Key=key_id, Body=json.dumps(event)) 

        #connect to dynamo
        table = dynamodb.Table('hw5') 
        
        #send to dynamodb
        try:
            response = table.get_item(
                Key={'user_id': event["user_id"]}
                )
            count=int(response['Item']['submission_count'])
        except KeyError:
            count=0
        
        table.put_item(
        Item={
            'user_id': event["user_id"],
            'timestamp': event["timestamp"],
            'time_elapsed': event["time_elapsed"],
            'q1': event["q1"],
            'q2': event["q2"],
            'q3': event["q3"],
            'q4': event["q4"],
            'q5': event["q5"],
            'freetext': event["freetext"],
            'submission_count': count+1,
            'source': source
            }
        )

        return {'statusCode': 200,'body': json.dumps('Valid Data: Data Loaded')}
    else:
        return {'statusCode': 400,'body': json.dumps('Invalid Data')}
