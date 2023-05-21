import logging
from bisect import bisect
from typing import Dict, Optional
import pandas as pd
import numpy as np

from .core import topic_segmentation
from .dataset import ami_dataset, icsi_dataset, add_durations, preprocessing
from .types import TopicSegmentationAlgorithm, TopicSegmentationDatasets
from nltk.metrics.segmentation import pk, windowdiff


MEETING_ID_COL_NAME = "meeting_id"
START_COL_NAME = "start_time"
END_COL_NAME = "end_time"
CAPTION_COL_NAME = "caption"
LABEL_COL_NAME = "label"


def compute_metrics(prediction_segmentations, binary_labels, metric_name_suffix="", verbose=False):
    #print(prediction_segmentations)
    #print(binary_labels)
    _pk, _windiff = [], []
    for meeting_id, reference_segmentation in binary_labels.items():

        predicted_segmentation_indexes = prediction_segmentations[meeting_id]
        # we need to convert from topic changes indexes to topic changes binaries
        predicted_segmentation = [0] * len(reference_segmentation)
        for topic_change_index in predicted_segmentation_indexes:
            predicted_segmentation[topic_change_index] = 1

        reference_segmentation = "".join(map(str, reference_segmentation))
        predicted_segmentation = "".join(map(str, predicted_segmentation))

        try:
            _pk.append(pk(reference_segmentation, predicted_segmentation))
        except ZeroDivisionError:
            _pk.append(np.nan)  # TODO: replace with correct solution

        # setting k to default value used in CoAP (pk) function for both evaluation functions
        try:
            k = int(round(len(reference_segmentation) / (reference_segmentation.count("1") * 2.0)))
            _windiff.append(windowdiff(reference_segmentation, predicted_segmentation, k))
        except ZeroDivisionError:
            _windiff.append(np.nan)  # TODO: replace with correct solution

    avg_pk = sum(_pk) / len(binary_labels)
    avg_windiff = sum(_windiff) / len(binary_labels)

    if verbose:
        print("Pk on {} meetings: {}".format(len(binary_labels), avg_pk))
        print("WinDiff on {} meetings: {}".format(len(binary_labels), avg_windiff))

    return {
        "average_Pk_" + str(metric_name_suffix): avg_pk,
        "average_windiff_" + str(metric_name_suffix): avg_windiff,
    }


def binary_labels_flattened(
    input_df,
    labels_df,
    meeting_id_col_name: str,
    start_col_name: str):
    """
    Binary Label [0, 0, 1, 0] for topic changes as ntlk format.
    Hierarchical topic strutcure flattened.
    see https://www.XXXX.com/intern/anp/view/?id=434543
    """
    labels_flattened = {}
    meeting_ids = list(set(input_df[meeting_id_col_name]))

    for meeting_id in meeting_ids:
        logging.info("\n\nMEETING ID:{}".format(meeting_id))

        if meeting_id not in list(labels_df[meeting_id_col_name]):
            logging.info("{} not found in `labels_df`".format(meeting_id))
            continue

        meeting_data = input_df[
            input_df[meeting_id_col_name] == meeting_id
        ].sort_values(by=[start_col_name])
        meeting_sentences = [*map(lambda s: s.lower(), list(meeting_data["caption"]))]

        caption_start_times = list(meeting_data[start_col_name])
        segment_start_times = list(
            labels_df[labels_df[meeting_id_col_name] == meeting_id][start_col_name]
        )

        meeting_labels_flattened = [0] * len(caption_start_times)

        # we skip first and last labaled segment cause they are naive segments
        for sst in segment_start_times[1:]:
            try:
                topic_change_index = caption_start_times.index(sst)
            except ValueError:
                topic_change_index = bisect(caption_start_times, sst)
                if topic_change_index == len(meeting_labels_flattened):
                    topic_change_index -= 1  # bisect my go out of boundary
            meeting_labels_flattened[topic_change_index] = 1

        labels_flattened[meeting_id] = meeting_labels_flattened

        logging.info("MEETING TRANSCRIPTS")
        for i, sentence in enumerate(meeting_sentences):
            if meeting_labels_flattened[i] == 1:
                logging.warning("\n\n<<------ Topic Change () ------>>\n")
            logging.info(sentence)

    return labels_flattened


def binary_labels_top_level(
    input_df,
    labels_df,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str):
    """
    Binary Label [0, 0, 1, 0] for topic changes as ntlk format.
    Hierarchical topic strutcure only top level topics
    see https://www.XXXX.com/intern/anp/view/?id=434543
    """
    labels_top_level = {}
    meeting_ids = list(set(input_df[meeting_id_col_name]))

    for meeting_id in meeting_ids:
        logging.info("\n\nMEETING ID:{}".format(meeting_id))

        if meeting_id not in list(labels_df[meeting_id_col_name]):
            logging.info("{} not found in `labels_df`".format(meeting_id))
            continue

        meeting_data = input_df[
            input_df[meeting_id_col_name] == meeting_id
        ].sort_values(by=[start_col_name])
        meeting_sentences = [*map(lambda s: s.lower(), list(meeting_data["caption"]))]

        caption_start_times = list(meeting_data[start_col_name])
        segment_start_times = list(
            labels_df[labels_df[meeting_id_col_name] == meeting_id][start_col_name]
        )
        segment_end_times = list(
            labels_df[labels_df[meeting_id_col_name] == meeting_id][end_col_name]
        )

        meeting_labels_top_level = [0] * len(caption_start_times)

        high_level_topics_indexes = []
        i = 0
        while i < len(segment_end_times):
            end = segment_end_times[i]
            high_level_topics_indexes.append(i)
            if segment_end_times.count(end) == 2:
                # skip all the subtopics of this high level topic
                i = (
                    segment_end_times.index(end)
                    + segment_end_times[segment_end_times.index(end) + 1 :].index(end)
                    + 2
                )
            else:
                i += 1

        segment_start_times_high_level = [
            segment_start_times[i] for i in high_level_topics_indexes
        ]

        # we skip first and last labaled segment cause they are naive segments
        for sst in segment_start_times_high_level[1:]:
            try:
                topic_change_index = caption_start_times.index(sst)
            except ValueError:
                topic_change_index = bisect(caption_start_times, sst)
                if topic_change_index == len(meeting_labels_top_level):
                    topic_change_index -= 1  # bisect my go out of boundary
            meeting_labels_top_level[topic_change_index] = 1

        labels_top_level[meeting_id] = meeting_labels_top_level

        logging.info("MEETING TRANSCRIPTS")
        for i, sentence in enumerate(meeting_sentences):
            if meeting_labels_top_level[i] == 1:
                logging.warning("\n\n<<------ Topic Change () ------>>\n")
            logging.info(sentence)

    return labels_top_level


def recode_labels(input_df,meeting_id_col_name,label_col_name):
    """
    Dictionary of meeting_id: reference_segmentation where latter is
    0/1 binary for transition sentences - for example, recode [1,1,2,2,2,3] to [0,0,1,0,0,0,1]
    """
    output = dict()
    for meeting_id in input_df[meeting_id_col_name].unique():
        labels = input_df[input_df[meeting_id_col_name]==meeting_id][label_col_name].to_list()
        recoded_labels = [0] * len(labels)
        current_label = labels[0]
        for i in range(len(labels)):
            if labels[i] != current_label:
                current_label = labels[i]
                recoded_labels[i] = 1
        output[meeting_id] = recoded_labels
    return output


def merge_metrics(*metrics):
    res = {}
    for m in metrics:
        for k, v in m.items():
            res[k] = v
    return res


def eval_topic_segmentation(
    topic_segmentation_algorithm: TopicSegmentationAlgorithm,
    dataset_name: Optional[TopicSegmentationDatasets] = None,
    input_df: Optional[pd.DataFrame] = None,
    col_names: Optional[tuple] = None,
    binary_label_encoding: Optional[bool] = False,
    return_segmentation: Optional[bool] = False,
    verbose: Optional[bool] = False
) -> Dict[str, float]:
    
    if dataset_name is not None:
        if dataset_name == TopicSegmentationDatasets.AMI:
            input_df, label_df = ami_dataset()
        elif dataset_name == TopicSegmentationDatasets.ICSI:
            input_df, label_df = icsi_dataset()
        elif dataset_name == TopicSegmentationDatasets.TEST:
            raise NotImplementedError("Test dataset not implemented yet.")
            input_df, label_df = test_video_dataset()  # unclear what this is referring to
        else:
            raise ValueError("Unknown dataset name.")
    
    if col_names is None:
        col_names = (MEETING_ID_COL_NAME,START_COL_NAME,END_COL_NAME,CAPTION_COL_NAME,LABEL_COL_NAME)
    meeting_id_col_name, start_col_name, end_col_name, caption_col_name, label_col_name = col_names

    prediction_segmentations = topic_segmentation(
        topic_segmentation_algorithm,input_df,meeting_id_col_name,
        start_col_name,end_col_name,caption_col_name,verbose=verbose)

    if dataset_name is not None:
        flattened = binary_labels_flattened(
            input_df,label_df,MEETING_ID_COL_NAME,START_COL_NAME)
        top_level = binary_labels_top_level(
            input_df,label_df,MEETING_ID_COL_NAME,START_COL_NAME,END_COL_NAME)
        flattened_metrics = compute_metrics(
            prediction_segmentations, flattened, metric_name_suffix="flattened",verbose=verbose)
        top_level_metrics = compute_metrics(
            prediction_segmentations, top_level, metric_name_suffix="top_level",verbose=verbose)
        return merge_metrics(flattened_metrics, top_level_metrics)
    else:
        if binary_label_encoding:
            labels = {}
            for meeting_id in input_df[meeting_id_col_name].unique():
                labels[meeting_id] = input_df[input_df[meeting_id_col_name]==meeting_id][label_col_name].to_list()
        else:
            labels = recode_labels(input_df,meeting_id_col_name,label_col_name)

        if return_segmentation:
            return compute_metrics(prediction_segmentations,labels,verbose=verbose), prediction_segmentations
        else:
            return compute_metrics(prediction_segmentations,labels,verbose=verbose)
    
    
def multiple_eval(
        data_function,iterations,test_algorithm,even_algorithm,random_algorithm,verbose=False,embeddings=False):

    #test_transcripts = []
    #segmentations = []

    n_captions = []
    n_segments = []
    metrics = []

    for i in range(iterations):

        if embeddings:
            results,embedding,labels,topics,doc_count = data_function(embeddings=embeddings)
        else:
            results,labels,topics,doc_count = data_function(embeddings=embeddings)
            embedding = None
        n_captions.append(len(results))
        n_segments.append(doc_count)
        test_data = pd.DataFrame(data={'caption':results,'label':labels,'meeting_id':1,'embedding':embedding})
        test_data = add_durations(test_data)
        test_data = test_data[['meeting_id','start_time','end_time','caption','label','embedding']]
        test_data = preprocessing(test_data, 'caption')

        metrics.append(
            [eval_topic_segmentation(
                topic_segmentation_algorithm=test_algorithm,input_df=test_data,verbose=verbose),
            eval_topic_segmentation(
                topic_segmentation_algorithm=even_algorithm,input_df=test_data,verbose=verbose),
            eval_topic_segmentation(
                topic_segmentation_algorithm=random_algorithm,input_df=test_data,verbose=verbose)])
        
    flattened_list = [{f'dict{i+1}_{k}': v for i, d in enumerate(trial) for k, v in d.items()} for trial in metrics]
    output = pd.DataFrame(flattened_list)
    output.columns = ['test_pk','test_windiff','even_pk','even_windiff','random_pk','random_windiff']
    output['n_captions'] = n_captions
    output['n_segments'] = n_segments
        
    return output