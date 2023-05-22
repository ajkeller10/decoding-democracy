import pickle
import pandas as pd
import random
import numpy as np
import itertools
import json
import os
from unsupervised_topic_segmentation import dataset
import ast
import torch


def transcript_pickle_to_df():
    '''
    load transcript pickle into data frame
    '''
    # import data 
    with open("data/transcripts.pickle", "rb") as f:
        transcripts = pickle.load(f)

    t = pd.DataFrame(transcripts.keys(),columns=["transcript_id"])
    t['sentences']=t["transcript_id"].apply(lambda x: transcripts[x])
    return t 

def snappy_parquet_to_df(path):
    '''
    load snappy parquet to data frame with transcript per row
    '''

    df = pd.read_parquet(path)
    df['embeddings'] = df['finished_embeddings_str'].apply(lambda x: torch.FloatTensor(list(ast.literal_eval(x))[0]))

    if 'topic_count' in df.columns: 
        return df.groupby(['transcript_id'], as_index = False).agg({'sentences':lambda x: list(x),
                                                        'topic_count':lambda x: list(x),
                                                        'topic_desc':lambda x: list(x),
                                                        'has_topic_desc':lambda x: list(x),
                                                        'embeddings':lambda x: list(x)
                                                        })
    else: 
        return df.groupby(['transcript_id'], as_index = False).agg({'sentences':lambda x: list(x),
                                                        'embeddings':lambda x: list(x)
                                                        })



def jsons_to_dict_list(topic_path):
    '''
    load json transcripts into list of transcripts
    '''
    dict_list=[]

    file_paths = os.listdir(topic_path)
    file_paths = list(filter(lambda x: x.endswith(".json"),file_paths))

    for file_path in file_paths:    
        with open( topic_path + file_path ) as file:
            json_load = json.loads(file.read())
        dict_list.append(json_load)

    return dict_list

def clean_topic_json(topic_json,transcript_id,fillers):
    '''
    clean list of topic jsons and combine results into cleaned df

    Args:
        topic_json: list of topic dictionaries containing text
        filler: list of filler words to be excluded in cleaning 
    
    Returns:
        df_clean: df with all sentences and corresponding topics post cleaning
    '''
    df_temp=pd.DataFrame()
    has_topic_desc=len(set([x['topic'] for x in topic_json]))>1

    for index, topic in enumerate(topic_json): 
        text = [x['text'] for x in topic['dialogueacts']]
        if df_temp.empty:
            df_temp=pd.DataFrame({'transcript_id':transcript_id,'sentences':text,'topic_count':index,'topic_desc':topic['topic'],'has_topic_desc':has_topic_desc})
        else:
            df_temp=pd.concat([df_temp,pd.DataFrame({'transcript_id':transcript_id,'sentences':text,'topic_count':index,'topic_desc':topic['topic'],'has_topic_desc':has_topic_desc})])    

    df_clean=dataset.preprocessing(df_temp,'sentences',fillers.copy(),min_caption_len=20)
    
    return df_clean


def generate_segment(
        t : pd.DataFrame = transcript_pickle_to_df(), doc_count_min:int =1,
        doc_count_max: int = 10, sentence_min: int = 20, labeled: bool = False, embeddings = False) -> tuple: 
    '''
    Generate testing transcript from transcript data, where output resembles a transcript but incorporates segments from 1:doc_count_max documents

    Output is representative of transcripts in the following ways: 
    - Sample of transcripts is filtered to documents of sufficient length (>=doc_count_max*sentence_min)
        - For this reason, doc_count_max<= 10 and sentence_min<= 50 are recommended
    - Documents used in output are randomly selected from remaining sample
    - Output length is drawn from sample distribution of remaining document lengths 
    - Output starts with text from the start of a document, and ends with text from the end of a document
        - If document count of 1 is selected, the output will be identical to a randomly selected text
        - If document count above 2 is selected, all other documents have sentences sampled from a random starting point within the document. Range of (0 : length - sentences_to_use)
    - Number of sentences selected from a document is independent of length of the document
        - Except if sentences selected surpasses the end of the document, in which case number of sentences ends with end of document
        - This approach lets document selection and starting point selection remain random
        - This exception occurs less frequently with larger document counts

    
    Args:
        t: transcripts as rows in df, containing columns: 
            'sentences': list of sentences in transcript
            'topic_counts': list of distinct topic per sentence in transcript (only needed if labeled) 
        doc_count_max: maximum number of documents to pull from. Actual number in range of 1:doc_count_max
        sentence_min: minimum number of sentences per document selected
        labeled: has labeled topic codings
    
    Returns:
        segment (list): List of sentences
        labels (list): Count of distinct document each sentence was pulled from. Range of 1:doc_count_max   
        topics (list): Count of distinct topic associated with each sentence. None if not labeled. 
        doc_count (int): Count of documents in segment

    '''

    if not ('sentences' in t.columns):
        raise Exception("'sentences' column does not exist in df")
    
    #filter text for sufficiently long texts 
    t['length']=t['sentences'].apply(lambda x: len(x)) 
    t_long=t[t['length']>=(doc_count_max*sentence_min)]

    #count of documents
    doc_count = random.randint(doc_count_min,doc_count_max)

    #pull documents based on document count
    text = t_long.sample(doc_count)

    #add row number as index
    text.reset_index(inplace=True,drop=True)
    text.reset_index(inplace=True)
    
    if doc_count>1:

        #get length of last document to be total length of new document
        #NOTE: this makes output length independent of doc_count
        length_sequence = text.iat[-1,int(text.columns.get_indexer(['length']))]

        #distribute sentences between documents - based on percentage of chosen length
        percents = np.random.randint(1,100, size=doc_count)
        text['sentences_to_use'] = length_sequence * percents / sum(percents)
        text['sentences_to_use']= text['sentences_to_use'].astype('int') + sentence_min

        #reallocate sentences from shorter documents to last documents if necessary
        #note: this lets us incorporate smaller documents, rather than oversampling from larger documents
        text['reallocate']=text['length']-text['sentences_to_use']
        text.loc[text['reallocate']>0,'reallocate'] = 0  
        text['sentences_to_use']=text['sentences_to_use']+text['reallocate']
        text['reallocate_cum']=text['reallocate'].cumsum()

        #decide where to pull the sentences from 
        text['sentences_start']=text.apply(lambda x: random.randint(0,x['length']-x['sentences_to_use']),axis=1)

        #the first segment start at 0, and the last end at -1
        text.iat[-1,int(text.columns.get_indexer(['sentences_start']))]=text.iloc[-1]['length']-text.iloc[-1]['sentences_to_use']+text.iloc[-1]['reallocate_cum']    
        text.iat[0,int(text.columns.get_indexer(['sentences_start']))]=0
    
    else:
        text['sentences_to_use'] = text['length'] 
        text['sentences_start']=0
       
    #get sentences
    text['results']=text.apply(lambda x: x['sentences'][x.sentences_start:(x.sentences_start+x.sentences_to_use)],axis=1)
    text['labels']=text.apply(lambda x: [x['index']] * x['sentences_to_use'],axis=1)
    if embeddings:
        text['embedding']=text.apply(lambda x: x['embedding'][x.sentences_start:(x.sentences_start+x.sentences_to_use)],axis=1)
    text['labels']=text.apply(lambda x: [x['index']] * x['sentences_to_use'],axis=1)
    
    results = list(itertools.chain.from_iterable(text['results']))
    labels = list(itertools.chain.from_iterable(text['labels']))
    
    if embeddings:
        embedding = list(itertools.chain.from_iterable(text['embedding']))
    else:
        embedding=None
        
    if labeled and 'topic_counts' in text.columns:
        text['topics']=text.apply(lambda x: x['topic_counts'][x.sentences_start:(x.sentences_start+x.sentences_to_use)],axis=1)
        text['labels_scaled']=text.apply(lambda x: [1000*x['index']] * x['sentences_to_use'],axis=1)
        text['topics']=text.apply(lambda x: [sum(y) for y in zip(x['topics'],x['labels_scaled'])], axis=1)
        topics = list(itertools.chain.from_iterable(text['topics']))
    else:
        topics = None

    if embeddings:
        return results, embedding, labels, topics, doc_count
    else:
        return results, labels, topics, doc_count

def try_create_test_data():
    t=transcript_pickle_to_df()
    hold=generate_segment(t, doc_count_max = 10, sentence_min = 20)  
    print(hold[2])
