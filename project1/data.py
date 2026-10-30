"""
Data loading for the ISEAR emotion-classification dataset.

ISEAR (gsri-18/ISEAR-dataset-complete) has:
  - column 'content': the text (a short account of an emotional experience)
  - column 'emotion': the label as a string (one of 7 emotions)
  - column 'Unnamed: 2': junk, ignored
  - a single split ('train'), so we create our own train/test split here

Hugging Face: https://huggingface.co/datasets/gsri-18/ISEAR-dataset-complete/viewer/default/train?p=75
"""

from datasets import load_dataset
from part1.clslib import NamedClasses

EMOTIONS = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

def load_data(test_size=0.2, seed=0):

    raw = load_dataset("gsri-18/ISEAR-dataset-complete")["train"]

    def to_text_label(example):
      return {
          "text": example["content"],
          "label": EMOTIONS.index(example["emotion"]), # integer label
      }

    # rebuild columns as text/label, dropping everything else (incl. 'Unnamed: 2')
    dataset = raw.map(to_text_label, remove_columns=raw.column_names)

    # create train/split
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    # keeps the order of labels
    outcome_space = NamedClasses(EMOTIONS)
    
    return dataset, outcome_space