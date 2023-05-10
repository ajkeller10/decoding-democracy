from random import random
import pandas as pd


def topic_segmentation_random(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    random_threshold: float = 0.9):

    segments = {}
    for meeting_id in set(df[meeting_id_col_name]):
        random_segmentation = []
        for i in range(sum(df[meeting_id_col_name] == meeting_id)):
            if random() > random_threshold:
                random_segmentation.append(i)
        print(f"Random segmentation: {random_segmentation}")
        segments[meeting_id] = random_segmentation
    return segments


def topic_segmentation_even(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    k: int):

    segments = {}
    for meeting_id in set(df[meeting_id_col_name]):
        even_segmentation = []
        for i in range(sum(df[meeting_id_col_name] == meeting_id)):
            if i % k == 0:
                even_segmentation.append(i)
        print(f"Even segmentation: {even_segmentation}")
        segments[meeting_id] = even_segmentation
    return segments