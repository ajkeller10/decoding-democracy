import pickle
import pandas as pd
import random
import numpy as np
import itertools

def transcript_pickle_to_pd():
    '''
    load transcript pickle into data frame
    '''
    # import data 
    with open("transcripts.pickle", "rb") as f:
        transcripts = pickle.load(f)

    t = pd.DataFrame(transcripts.keys(),columns=["transcript_id"])
    t['sentences']=t["transcript_id"].apply(lambda x: transcripts[x])
    t['length']=t['sentences'].apply(lambda x: len(x)) 

    return t 

def generate_segment(t : pd.DataFrame, doc_count_limit: int = 10, sentence_min: int = 20) -> tuple: 
    '''
    Generate testing transcript from transcript data, where output resembles a transcript but incorporates segments from 1:doc_count_limit documents

    Output is representative of transcripts in the following ways: 
    - Sample of transcripts is filtered to documents of sufficient length (>=doc_count_limit*sentence_min)
        - For this reason, doc_count_limit<= 10 and sentence_min<= 50 are recommended
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
        t: transcripts
        doc_count_limit: maximum number of documents to pull from. Actual number in range of 1:doc_count_limit
        sentence_min: minimum number of sentences per document selected
    
    Returns:
        segment (list): List of sentences
        labels (list): Index of which document each sentence was pulled from. Range of 1:doc_count_limit         
    '''

    #filter text for sufficiently long texts 
    t_long=t[t['length']>=(doc_count_limit*sentence_min)]

    #count of documents
    doc_count = random.randint(1,doc_count_limit)
    
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
        text.loc[text['reallocate']>0,'reallocate'] = 0  ### DK: I think this always sets reallocate to 0 so below steps have no effect?
        text['sentences_to_use']=text['sentences_to_use']+text['reallocate']
        text['reallocate_cum']=text['reallocate'].cumsum()

        #decide where to pull the sentences from 
        text['sentences_start']=text.apply(lambda x: random.randint(0,x['length']-x['sentences_to_use']),axis=1)

        #the first segment start at 0, and the last end at -1
        text.iat[-1,int(text.columns.get_indexer(['sentences_start']))]=text.iloc[-1]['length']-text.iloc[-1]['sentences_to_use']+text.iloc[-1]['reallocate_cum']    
        #text.loc[text['sentences_start']<0,'sentences_start'] = 0 #prevent overflow: should only needed when doc_count=1
        text.iat[0,int(text.columns.get_indexer(['sentences_start']))]=0
    
    else: 
        text['sentences_to_use'] = text['length'] 
        text['sentences_start']=0
       
    #get sentences
    text['results']=text.apply(lambda x: x['sentences'][x.sentences_start:(x.sentences_start+x.sentences_to_use)],axis=1) #)
    text['labels']=text.apply(lambda x: [x['index']] * x['sentences_to_use'],axis=1)
    
    results = list(itertools.chain.from_iterable(text['results']))
    labels = list(itertools.chain.from_iterable(text['labels']))

    return results,labels,doc_count

def try_create_test_data():
    t=transcript_pickle_to_pd()
    hold=generate_segment(t, doc_count_limit = 10, sentence_min = 20)  
    print(hold[2])


