#informed by https://segments.ai/blog/pytorch-on-lambda
import torch
from transformers import RobertaConfig, RobertaModel, AutoTokenizer, AutoModel

configuration = RobertaConfig()
roberta_model = RobertaModel(configuration)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model_new = AutoModel.from_pretrained("roberta-base")

traced_model.save('resnet34.pt')