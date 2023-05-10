#!/usr/bin/env python3

from enum import Enum
from typing import NamedTuple, Optional

class TopicSegmentationDatasets(Enum):
    AMI = 0
    ICSI = 1
    TEST = 2

class TopicSegmentationAlgorithm():
    ID = ''

class RandomSegmentation(TopicSegmentationAlgorithm):
    def __init__(self,random_threshold: float = 0.9):
        self.RANDOM_THRESHOLD = random_threshold
        self.ID = 'random'

    def __repr__(self):
        return "RandomSegmentation(" + \
            "RANDOM_THRESHOLD=" + str(self.RANDOM_THRESHOLD) + ")"

class EvenSegmentation(TopicSegmentationAlgorithm):
    def __init__(self,k: int = 30):
        self.k = k
        self.ID = 'even'

    def __repr__(self):
        return "EvenSegmentation(" + \
            "k=" + str(self.k) + ")"
    
class TextTilingHyperparameters():
    ID = ''

class OriginalSegmentation(TextTilingHyperparameters):
    def __init__(
            self, smoothing_passes: int = 2,
            smoothing_window: int = 1, 
            topic_change_threshold: float = 0.6,
            max_segments_cap: bool = True, 
            max_segments_cap__average_segment_length: int = 60):
        self.SMOOTHING_PASSES = smoothing_passes
        self.SMOOTHING_WINDOW = smoothing_window
        self.TOPIC_CHANGE_THRESHOLD = topic_change_threshold
        self.MAX_SEGMENTS_CAP = max_segments_cap
        self.MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH = max_segments_cap__average_segment_length
        self.ID = 'original_segmentation'

    def __repr__(self):
        return "OriginalSegmentation(" + \
            "SMOOTHING_PASSES=" + str(self.SMOOTHING_PASSES) + ", " + \
            "SMOOTHING_WINDOW=" + str(self.SMOOTHING_WINDOW) + ", " + \
            "TOPIC_CHANGE_THRESHOLD=" + str(self.TOPIC_CHANGE_THRESHOLD) + \
            "MAX_SEGMENTS_CAP=" + str(self.MAX_SEGMENTS_CAP) + ", " + \
            "MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH=" + str(self.MAX_SEGMENTS_CAP__AVERAGE_SEGMENT_LENGTH) + ")"
    
class NewSegmentation(TextTilingHyperparameters):
    def __init__(self, stdevs: float = 1):
        self.STDEVS = stdevs
        self.ID = 'new_segmentation'

    def __repr__(self):
        return "NewSegmentation(" + \
            "STDEVS=" + str(self.STDEVS) + ")"

class BERTSegmentation(TopicSegmentationAlgorithm):
    def __init__(
        self, sentence_comparison_window: int = 15, 
        text_tiling: Optional[TextTilingHyperparameters] = None):
        self.SENTENCE_COMPARISON_WINDOW = sentence_comparison_window
        self.TEXT_TILING = text_tiling
        self.ID = 'bert'

    def __repr__(self):
        return "BERTSegmentation(" + \
            "SENTENCE_COMPARISON_WINDOW=" + str(self.SENTENCE_COMPARISON_WINDOW) + ", " + \
            "TEXT_TILING=" + str(self.TEXT_TILING) + ")"

class SBERTSegmentation():
    def __init__(self):
        self.ID = 'sbert'
        raise NotImplementedError()