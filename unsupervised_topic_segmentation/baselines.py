from random import random
import pandas as pd


SPLIT_VOCAB  = ['agenda']


def topic_segmentation_random(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    random_threshold: float = 0.9,
    verbose: bool = False):

    segments = {}
    for meeting_id in set(df[meeting_id_col_name]):
        random_segmentation = []
        for i in range(sum(df[meeting_id_col_name] == meeting_id)):
            if random() > random_threshold:
                random_segmentation.append(i)
        if verbose:
            print(f"Random segmentation: {random_segmentation}")
        segments[meeting_id] = random_segmentation
    return segments


def topic_segmentation_even(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    k: int,
    verbose: bool = False):

    segments = {}
    for meeting_id in set(df[meeting_id_col_name]):
        even_segmentation = []
        for i in range(sum(df[meeting_id_col_name] == meeting_id)):
            if i % k == 0:
                even_segmentation.append(i)
        if verbose:
            print(f"Even segmentation: {even_segmentation}")
        segments[meeting_id] = even_segmentation
    return segments


def topic_segmentation_none(
    df: pd.DataFrame,
    meeting_id_col_name: str):

    return {[] for _ in set(df[meeting_id_col_name])}


def topic_segmentation_lexical(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    caption_col_name: str = 'caption',
    split_vocabulary=SPLIT_VOCAB,
    verbose: bool = False):
    
    segments = {}
    for meeting_id in set(df[meeting_id_col_name]):
        lexical_segmentation = []
        for i in range(sum(df[meeting_id_col_name] == meeting_id)):
            if df[df[meeting_id_col_name] == meeting_id][caption_col_name].iloc[
                i].str.contains(split_vocabulary).any():
                lexical_segmentation.append(i)
        if verbose:
            print(f"Lexical segmentation: {lexical_segmentation}")
        segments[meeting_id] = lexical_segmentation
    return segments