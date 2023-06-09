"""TextTiling with BERT embeddings algorithm implementation."""


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import torch
from . import baselines as topic_segmentation_baselines
from .types import TopicSegmentationAlgorithm, BERTSegmentation, TextTilingHyperparameters
from transformers import RobertaConfig, RobertaModel, AutoTokenizer, AutoModel
from typing import Optional


# pretrained Roberta model
configuration = RobertaConfig()
roberta_model = RobertaModel(configuration)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model_new = AutoModel.from_pretrained("roberta-base")


PARALLEL_INFERENCE_INSTANCES = 20
DISPLAY_SIMILARITIES = False


def depth_score(timeseries: list[float])-> list[float]:
    """
    The depth score corresponds to how strongly the cues for a subtopic changed on both sides of a
    given token-sequence gap and is based on the distance from the peaks on both sides of the valley to that valley.

    (Only used in Solbiati et al's OriginalSegmentation)
    """
    depth_scores = []
    for i in range(1, len(timeseries) - 1):
        left, right = i - 1, i + 1
        while left > 0 and timeseries[left - 1] > timeseries[left]:
            left -= 1
        while (
            right < (len(timeseries) - 1) and timeseries[right + 1] > timeseries[right]
        ):
            right += 1
        depth_scores.append(
            (timeseries[right] - timeseries[i]) + (timeseries[left] - timeseries[i])
        )
    return depth_scores


def smooth(
        timeseries: list[float],
        n: int,
        s: int) -> list[float]:
    """Smooths time series. Only used in Solbiati et al's OriginalSegmentation."""
    smoothed_timeseries = timeseries[:]
    for _ in range(n):
        for index in range(len(smoothed_timeseries)):
            neighbours = smoothed_timeseries[
                max(0, index - s) : min(len(timeseries) - 1, index + s)
            ]
            smoothed_timeseries[index] = sum(neighbours) / len(neighbours)
    return smoothed_timeseries


def sentences_similarity(
        first_sentence_features: torch.Tensor, 
        second_sentence_features: torch.Tensor) -> float:
    """
    Given two embedding features compute cosine similarity. 
    """
    similarity_metric = torch.nn.CosineSimilarity()
    return float(similarity_metric(first_sentence_features, second_sentence_features))


def compute_window(
        timeseries: list[float],
        start_index: int,
        end_index: int):
    """Given start and end index of embedding, compute pooled window value.

    [window_size, 768] -> [1, 768]
    """
    with torch.no_grad():
        stack = torch.stack([features[0] for features in timeseries[start_index:end_index]])
        stack = stack.unsqueeze(
            0
        )  # https://jbencook.com/adding-a-dimension-to-a-tensor-in-pytorch/
        stack_size = end_index - start_index
        pooling = torch.nn.MaxPool2d((stack_size - 1, 1))
        return pooling(stack)


def block_comparison_score(
        timeseries: list[float],
        k: int)-> list[float]:
    """
    comparison score for a gap (i)

    cfr. docstring of block_comparison_score
    """
    res = []
    for i in range(k, len(timeseries) - k):  #  need window cushion on each end
        first_window_features = compute_window(timeseries, i - k, i + 1)
        second_window_features = compute_window(timeseries, i + 1, i + k + 2)
        res.append(
            sentences_similarity(first_window_features[0], second_window_features[0])
        )

    return res


def get_features_from_sentence(
        batch_sentences: list[str],
        layer: Optional[int] = -2,
        old_version: Optional[bool] = False) -> list[torch.Tensor]:
    """
    extracts the BERT semantic representation
    from a sentence, using an averaged value of
    the `layer`-th layer

    returns a 1-dimensional tensor of size 758 [old] or 768 [new]
    """
    batch_features = []
    if old_version:  # Solbiati et al's code - old version of transformers, unclear which one
        for sentence in batch_sentences:
            tokens = roberta_model.encode(sentence)
            all_layers = roberta_model.extract_features(tokens, return_all_hiddens=True)
            pooling = torch.nn.AvgPool2d((len(tokens), 1))
            sentence_features = pooling(all_layers[layer])
            batch_features.append(sentence_features[0])            
    else:  # rewritten to (hopefully) do the same thing
        for sentence in batch_sentences:
            with torch.no_grad():
                tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
                # attention mask usually just all ones
                input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
                all_layers = roberta_model_new(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states  # 13 layers
                pooling = torch.nn.AvgPool2d((input_ids.size(1), 1))
                sentence_features = pooling(all_layers[layer])  # 1 x n_tokens x 768
            batch_features.append(sentence_features[0])            
    return batch_features


def arsort2(array1, array2):
    """Helper function for Solbiati et al's OriginalSegmentation."""
    x = np.array(array1)
    y = np.array(array2)

    sorted_idx = x.argsort()[::-1]
    return x[sorted_idx], y[sorted_idx]


def get_local_maxima(array):
    """Gets ALL local maxima of an array. Only used in Solbiati et al's OriginalSegmentation."""
    local_maxima_indices = []
    local_maxima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            local_maxima_indices.append(i)
            local_maxima_values.append(array[i])
    return local_maxima_indices, local_maxima_values


def depth_score_to_topic_change_indexes(
    depth_score_timeseries: list[float],
    meeting_duration: int,
    topic_segmentation_configs: TextTilingHyperparameters
    )-> list[int]:
    """
    capped add a max segment limit so there are not too many segments, used for UI improvements on the Workplace TeamWork product

    Only used in Solbiati et al's OriginalSegmentation.
    """

    capped = topic_segmentation_configs.TEXT_TILING.MAX_SEGMENTS_CAP
    average_segment_length = (
        topic_segmentation_configs.TEXT_TILING.MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH
    )
    threshold = topic_segmentation_configs.TEXT_TILING.TOPIC_CHANGE_THRESHOLD * max(
        depth_score_timeseries)

    if depth_score_timeseries == []:
        return []

    local_maxima_indices, local_maxima = get_local_maxima(depth_score_timeseries)  # simple: higher than neighbors

    if local_maxima == []:
        return []

    if capped:  # capped is segmentation used for UI
        # sort based on maxima for pruning
        local_maxima, local_maxima_indices = arsort2(local_maxima, local_maxima_indices)

        # local maxima are sorted by depth_score value and we take only the first K
        # where the K+1th local maxima is lower then the threshold
        for thres in range(len(local_maxima)):
            if local_maxima[thres] <= threshold:
                break

        max_segments = int(meeting_duration / average_segment_length)
        slice_length = min(max_segments, thres)

        local_maxima_indices = local_maxima_indices[:slice_length]
        local_maxima = local_maxima[:slice_length]

        # after pruning, sort again based on indices for chronological ordering
        local_maxima_indices, _ = arsort2(local_maxima_indices, local_maxima)

    else:  # this is the vanilla TextTiling used for Pk optimization - just take local maxima above threshold
        filtered_local_maxima_indices = []
        filtered_local_maxima = []

        for i, m in enumerate(local_maxima):
            if m > threshold:
                filtered_local_maxima.append(m)
                filtered_local_maxima_indices.append(i)

        local_maxima = filtered_local_maxima
        local_maxima_indices = filtered_local_maxima_indices

    return local_maxima_indices


def get_timeseries(caption_indexes,features):
    timeseries = []
    for caption_index in caption_indexes:
        timeseries.append(features[caption_index])
    return timeseries


def flatten_features(batches_features):
    res = []
    for batch_features in batches_features:
        res += batch_features
    return res


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (
        a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(min(len(a), n)))


def statistical_segmentation(
        similarities: list[float],
        stdevs: int) -> tuple[list[int], float]:
    """Core method for NewSegmentation - finds local minima below a statistical threshold."""
    indices = []
    start = None
    threshold = np.mean(similarities)-(np.std(similarities)*stdevs)
    # specifically, get minimum of each sequence of consecutive scores below the threshold
    for i, value in enumerate(similarities):
        if value < threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                min_index = start + similarities[start:i].index(min(similarities[start:i]))
                indices.append(min_index)
                start = None
    return indices, threshold


def fix_indices(
        segments: list[float],
        window_size: int,
        segmenting_method: str) -> tuple[list[float], int]:
    """Original code incorrectly returned indices relative to original indexing. 
    
    This fixes that.
    """
    segments.sort()
    additions = window_size
    if segmenting_method=="original_segmentation":
        additions += 1
    return [segment+additions for segment in segments], additions


def topic_segmentation(
    topic_segmentation_algorithm: TopicSegmentationAlgorithm,
    df: pd.DataFrame,
    meeting_id_col_name: Optional[str] = "meeting_id",
    start_col_name: Optional[str] = "start",
    end_col_name: Optional[str] = "end",
    caption_col_name: Optional[str] = "caption",
    embedding_col_name: Optional[str] = "embedding",
    verbose: Optional[bool] = False,
    return_plot: Optional[bool] = False):
    """
    Core segmentation method wrapping any algorithm.
    """

    if topic_segmentation_algorithm.ID == "bert":
        return topic_segmentation_bert(
            topic_segmentation_algorithm,
            df,
            meeting_id_col_name,
            start_col_name,
            end_col_name,
            caption_col_name,
            embedding_col_name,
            verbose=verbose,
            return_plot=return_plot)
    elif topic_segmentation_algorithm.ID == "random":
        return topic_segmentation_baselines.topic_segmentation_random(
            df, meeting_id_col_name, topic_segmentation_algorithm.RANDOM_THRESHOLD, verbose=verbose)
    elif topic_segmentation_algorithm.ID == "even":
        return topic_segmentation_baselines.topic_segmentation_even(
            df, meeting_id_col_name, topic_segmentation_algorithm.k, verbose=verbose)
    elif topic_segmentation_algorithm.ID == "none":
        return topic_segmentation_baselines.topic_segmentation_none(
            df, meeting_id_col_name)
    elif topic_segmentation_algorithm.ID == "lexical":
        return topic_segmentation_baselines.topic_segmentation_lexical(
            df, meeting_id_col_name, caption_col_name,
            topic_segmentation_algorithm.SPLIT_VOCABULARY, verbose=verbose)
    else:
        raise NotImplementedError("Algorithm not implemented")


def topic_segmentation_bert(
    topic_segmentation_configs: BERTSegmentation,
    df: pd.DataFrame,
    meeting_id_col_name: str,
    start_col_name: Optional[str] = "start",
    end_col_name: Optional[str] = "end",
    caption_col_name: Optional[str] = "caption",
    embedding_col_name: Optional[str] = "embedding",
    verbose: Optional[bool] = False,
    return_plot: Optional[bool] = False):
    """Core segmentation method for TextTiling with embeddings."""

    textiling_hyperparameters = topic_segmentation_configs.TEXT_TILING

    if embedding_col_name not in df.columns:
        print('No embedding column found, running RoBERTa inference.')
        batches_features = []
        for batch_sentences in split_list(
            df[caption_col_name], PARALLEL_INFERENCE_INSTANCES
        ): # splits into sequential batches such that total number of batches equals INSTANCES value
            batches_features.append(get_features_from_sentence(batch_sentences)) # list of tensors of size (1,768), one for each sentence
        features = flatten_features(batches_features)   # changes back to list of length 768 tensors, one for each sentence in dataset
    else:
        features = list(df[embedding_col_name])

    segments = {}
    for meeting_id in set(df[meeting_id_col_name]):

        meeting_data = df[df[meeting_id_col_name] == meeting_id]
        caption_indexes = list(meeting_data.index)

        timeseries = get_timeseries(caption_indexes, features)  # this is just supposed to reset order according to index
        block_comparison_score_timeseries = block_comparison_score(
            timeseries, k=topic_segmentation_configs.SENTENCE_COMPARISON_WINDOW
        )  # this is list of length len(timeseries) - 2*SENTENCE_COMPARISON_WINDOW
        # each element is the similarity score of window before and after

        if textiling_hyperparameters.ID=="original_segmentation":

            # does some smoothing on list described above, small (<1%) effect in some tests
            block_comparison_score_timeseries = smooth(
                block_comparison_score_timeseries,
                n=textiling_hyperparameters.SMOOTHING_PASSES,
                s=textiling_hyperparameters.SMOOTHING_WINDOW)

            # "The depth score corresponds to how strongly the cues for a subtopic changed on both sides of a
            # given token-sequence gap and is based on the distance from the peaks on both sides of the valley to that valley."
            # what's weird is this seems like different approach than what was in paper?? is it somehow equivalent?
            depth_score_timeseries = depth_score(block_comparison_score_timeseries)
            # produces list of length of comparison scores minus 2

            meeting_start_time = meeting_data[start_col_name].iloc[0]
            meeting_end_time = meeting_data[end_col_name].iloc[-1]
            meeting_duration = meeting_end_time - meeting_start_time
            segments[meeting_id] = depth_score_to_topic_change_indexes(
                depth_score_timeseries,
                meeting_duration,
                topic_segmentation_configs=topic_segmentation_configs)
            
        elif textiling_hyperparameters.ID=="new_segmentation":  # new method is as described in paper

            segments[meeting_id], threshold = statistical_segmentation(
                block_comparison_score_timeseries, 
                stdevs = textiling_hyperparameters.STDEVS)
            
        else:
            raise NotImplementedError("TextTiling method not implemented")
            
        segments[meeting_id], additions = fix_indices(
            segments[meeting_id],
            topic_segmentation_configs.SENTENCE_COMPARISON_WINDOW,
            textiling_hyperparameters.ID)
        
        if verbose:
            print(f'BERT segmentation: {segments[meeting_id]}')
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            indexes = np.arange(0,len(block_comparison_score_timeseries),1) + additions
            plt.plot(indexes,block_comparison_score_timeseries)
            #plt.plot(
            #    indexes,
            #    [np.mean(block_comparison_score_timeseries)]*
            #    len(block_comparison_score_timeseries))
            plt.plot(
                indexes,
                [threshold]*
                len(block_comparison_score_timeseries))
            for i in segments[meeting_id]:
                plt.plot(i,block_comparison_score_timeseries[i-additions],'r*')
            plt.xlabel('Sentence Index')
            plt.ylabel('Similarity Score')
            plt.legend(['Similarity Score', 'Threshold', 'Predicted Transition'])

            if return_plot:
                return segments, fig, ax
            else:
                plt.show()

    return segments