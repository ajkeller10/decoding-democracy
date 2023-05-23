# Decoding Democracy: Segmentation of City Council Transcripts

This is the final project from [@ajkeller10](https://github.com/ajkeller10) and [@olpinney](https://github.com/olpinney) for CAPP 30255 Advanced Machine Learning for Public Policy in Spring 2023.

## Project Intro / Objective
Our project builds on the [Council Data Project](https://github.com/CouncilDataProject/cdp-roadmap/issues/9), an open-source product that makes local government meetings more accessible by automatically transcribing meeting recordings. Our project attempts to segment transcripts to improve usefulness –  an open to-do in Council Data Project’s backlog.

### Prior work utilized 
#### Unsupervised Topic Segmentation of Meetings with BERT Embeddings
- [original work](https://github.com/gdamaskinos/unsupervised_topic_segmentation/tree/main)
- [modified code in project repo](https://github.com/ajkeller10/decoding-democracy/tree/main/unsupervised_topic_segmentation)

#### AMI and ICSI Corpora
- [original work](https://github.com/guokan-shang/ami-and-icsi-corpora)
- [modified code in project repo](https://github.com/ajkeller10/decoding-democracy/tree/main/data/ami-and-icsi-corpora-master)

### English RoBERTa Embeddings
- [original work](https://sparknlp.org/2022/04/14/roberta_embeddings_distilroberta_base_en_3_0.html)

### Technologies
- Python
- PySpark

## Getting Started

### Environment Variables
- [environment file: build-agnostic](https://github.com/ajkeller10/decoding-democracy/blob/main/environment.yml)

### Spark provisioning
- all cloud resources used for this project have been terminated
- provisioned Apache Spark Cluster Architecture with m5.xlarge head for AMI embeddings. Larger head is needed for transcript embeddings
- loaded [parquet data files](https://github.com/ajkeller10/decoding-democracy/tree/main/data) to "decoding-democracy-embed" s3 bucket
- use of Spark is optional - embedding can be done [locally](https://github.com/ajkeller10/decoding-democracy/blob/main/unsupervised_topic_segmentation/core.py) but will slow down code dramatically

### Main analysis files
- [explanation of methods and exploration of manually labeled transcript data](https://github.com/ajkeller10/decoding-democracy/blob/main/demonstrate_segmentation_methods.ipynb)
- [evaluating random concatenation proxy task on AMI corpus](https://github.com/ajkeller10/decoding-democracy/blob/main/test_with_embeddings.ipynb)


## Approach

### Algorithm: TextTiling with transformers:
- Embed sentences using language model (ie average-pooled penultimate RoBERTa layer) 
- Calculate sequence of cosine similarities between pooled embeddings of length-k sliding window
- Segment at local minima when similarity is lower than j standard deviations from the mean

#### Links
- [link to Spark embeddings](https://github.com/ajkeller10/decoding-democracy/blob/main/data_cleaning.ipynb)
- [link to local methods](https://github.com/ajkeller10/decoding-democracy/blob/main/unsupervised_topic_segmentation/core.py)
  
![Table 1: TextTiling Results on Manually Labeled Transcript](https://github.com/ajkeller10/decoding-democracy/tree/main/tables/table1.png)

### Hyperparameter tuning:
- We needed a proxy task to tune hyperparameters k and j and evaluate performance for unlabeled transcripts
- Target task: segmenting concatenated transcripts
- We evaluated the transferability of this task via a labeled dataset: topic-annotated AMI Meeting Corpus

#### Links
- [link to workflow](https://github.com/ajkeller10/decoding-democracy/blob/main/test_with_embeddings.ipynb)

### Metrics:
- Compared Pk, WinDiff, and # segments to three baselines: segmenting evenly, randomly, and lexically (e.g. split segments on any mention of “agenda” or “next item”)

#### Links
- [link to evaluation methods](https://github.com/ajkeller10/decoding-democracy/blob/main/unsupervised_topic_segmentation/eval.py)


## Data: 
### All transcribed meetings from Milwaukee
- 412 transcripts, 330K sentences, 4.8M words
- Average 800 sentences/transcript, 15 words/sentence

#### Links
- [source data](https://github.com/CouncilDataProject/milwaukee)
- [pulling Source data](https://github.com/ajkeller10/decoding-democracy/blob/main/download_data.ipynb)
- [raw data](https://github.com/ajkeller10/decoding-democracy/blob/main/data/transcripts.pickle)
- [manually labeled data](https://github.com/ajkeller10/decoding-democracy/tree/main/data/manually_labeled)

### AMI meeting data
- 136 transcripts, 30k sentences, 0.3M words
- Average 220 sentences/transcript, 11 words/sentence

#### Links
- [source data](https://groups.inf.ed.ac.uk/ami/corpus/)
- [code for Pulling Source data](https://github.com/ajkeller10/decoding-democracy/blob/main/data/ami-and-icsi-corpora-master/ami-corpus/topics.py)
- [raw data](https://github.com/ajkeller10/decoding-democracy/tree/main/data/ami-and-icsi-corpora-master/ami-corpus/output/topics)
- [code for data cleaning pre- and post-Spark](https://github.com/ajkeller10/decoding-democracy/blob/main/data_cleaning.ipynb)
- [code for data embedding in Spark](https://github.com/ajkeller10/decoding-democracy/blob/main/spark_roberta_pipeline.ipynb)
- embedded data is 190MB and cannot be stored on github. Please contact for copy of "decoding-democracy/data/transcripts_with_embeds.pickle"
- [code for generating random concatenated transcripts](https://github.com/ajkeller10/decoding-democracy/blob/main/create_test_data.py)

## Results
### Takeaway 1: 
For AMI meeting transcript data, segmenting randomly concatenated documents does not appear to be a good proxy task for topic segmentation. 

#### Links
- [analysis file](https://github.com/ajkeller10/decoding-democracy/blob/main/test_with_embeddings.ipynb)

![Table 2: AMI Meeting Corpus Performance](https://github.com/ajkeller10/decoding-democracy/tree/main/tables/table2.JPG)

Concatenated topic segmentation suggests window k=70 and threshold j=1 to balance Pk and WinDiff, but this is suboptimal for true segmentation task.

### Takeaway 2: 
TextTiling with transformers is anecdotally successful at segmenting council transcripts; however, hyperparameter tuning and performance evaluation require manually-labeled data.

![Table 3: Manually Labeled Council Transcript Performance](https://github.com/ajkeller10/decoding-democracy/tree/main/tables/table3.png)

#### Links
- [analysis file](https://github.com/ajkeller10/decoding-democracy/blob/main/demonstrate_segmentation_methods.ipynb)




