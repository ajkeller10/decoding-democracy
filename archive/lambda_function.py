import json
import boto3
from transformers import RobertaConfig, RobertaModel, AutoTokenizer, AutoModel
import torch


# pretrained Roberta model
configuration = RobertaConfig()
roberta_model = RobertaModel(configuration)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model_new = AutoModel.from_pretrained("roberta-base")

def embed_sentence(sentence, layer=-2):
    """
    extracts the BERT semantic representation
    from a sentence, using an averaged value of
    the `layer`-th layer

    returns a 1-dimensional tensor of size 768
    """
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
    all_layers = roberta_model_new(
        input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
    pooling = torch.nn.AvgPool2d((input_ids.size(1), 1))
    sentence_features = pooling(all_layers[layer])
    return sentence_features[0] 

def lambda_handler(event, context):   

    transcripts = event['transcripts']

    #embeddings = [[embed_sentence(sentence, layer=-2) for sentence in sentences] for sentences in transcripts]
    return {'statusCode': 200,'body': json.dumps(event)}


           
