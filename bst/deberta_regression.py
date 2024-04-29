import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import warnings
# warnings.simplefilter('ignore')

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from tokenizers import AddedToken
import torch

USE_REGRESSION = True

VER=1

LOAD_FROM = None

COMPUTE_CV = True