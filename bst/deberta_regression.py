import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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

USE_REGRESSION = True

VER=3

LOAD_FROM = None

COMPUTE_CV = True


class PATHS:
    train_path = '../dataset/4k_nooverlap.csv'
    test_path = '../dataset/test.csv'
    submission_path = '../dataset/sample_submission.csv'
    model_dir = '/ai/users/bst/competition/model/microsoft/'
    model_name = 'deberta-v3-large'
    model_path = model_dir + model_name
    output_dir = '/ai/users/bst/competition/output_v{VER}'

class CFG:
    n_splits = 10
    seed = 42
    max_length = 1536
    lr = 1e-5
    train_batch_size = 1
    eval_batch_size = 1
    gradient_accumulation_steps = 4
    train_epochs = 4
    weight_decay = 0.01
    warmup_ratio = 0.05
    num_labels = 6

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(seed=CFG.seed)


class Tokenize(object):
    def __init__(self, train, valid, tokenizer):
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid
        
    def get_dataset(self, df):
        ds = Dataset.from_dict({
                'essay_id': [e for e in df['essay_id']],
                'full_text': [ft for ft in df['full_text']],
                'label': [s for s in df['label']],
            })
        return ds
        
    def tokenize_function(self, example):
        tokenized_inputs = self.tokenizer(
            example['full_text'], truncation=True, max_length=CFG.max_length
        )
        return tokenized_inputs
    
    def __call__(self):
        train_ds = self.get_dataset(self.train)
        valid_ds = self.get_dataset(self.valid)
        
        tokenized_train = train_ds.map(
            self.tokenize_function, batched=True
        )
        tokenized_valid = valid_ds.map(
            self.tokenize_function, batched=True
        )
        
        return tokenized_train, tokenized_valid, self.tokenizer


# 如果是回归任务，需要修改compute_metrics_for_regression函数
def compute_metrics_for_regression(eval_pred):
    
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.clip(0,5).round(0), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results

# 如果是分类任务，需要修改compute_metrics_for_classification函数
def compute_metrics_for_classification(eval_pred):
    
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights='quadratic')
    results = {
        'qwk': qwk
    }
    return results


data = pd.read_csv(PATHS.train_path)
data['label'] = data['score'].apply(lambda x: x-1)
if USE_REGRESSION: 
    data["label"] = data["label"].astype('float32') 
else: 
    data["label"] = data["label"].astype('int32') 

skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
for i, (_, val_index) in enumerate(skf.split(data, data["score"])):
    data.loc[val_index, "fold"] = i


training_args = TrainingArguments(
    output_dir=PATHS.output_dir,
    fp16=True,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    num_train_epochs=CFG.train_epochs,
    weight_decay=CFG.weight_decay,
    logging_steps=500,
    evaluation_strategy='steps',
    metric_for_best_model='qwk',
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=5,
    load_best_model_at_end=True,
    report_to='none',
    warmup_ratio=CFG.warmup_ratio,
    lr_scheduler_type='linear', # "cosine" or "linear" or "constant"
    optim='adamw_torch',
    logging_first_step=True,
    save_only_model=True,
)


if COMPUTE_CV:
    for fold in range(len(data['fold'].unique())):

        # GET TRAIN AND VALID DATA
        train = data[data['fold'] != fold]
        valid = data[data['fold'] == fold].copy()

        # ADD NEW TOKENS for ("\n") new paragraph and (" "*2) double space 
        tokenizer = AutoTokenizer.from_pretrained(PATHS.model_path)
        tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])
        tokenize = Tokenize(train, valid, tokenizer)
        tokenized_train, tokenized_valid, _ = tokenize()

        # REMOVE DROPOUT FROM REGRESSION
        config = AutoConfig.from_pretrained(PATHS.model_path)
        if USE_REGRESSION:
            config.attention_probs_dropout_prob = 0.0 
            config.hidden_dropout_prob = 0.0 
            config.num_labels = 1 
        else: config.num_labels = CFG.num_labels 

        if LOAD_FROM:
            model = AutoModelForSequenceClassification.from_pretrained( f'/ai/users/bst/competition/outputs/{PATHS.model_name}_AES2_fold_{fold}_v{VER}')
        else:
            model = AutoModelForSequenceClassification.from_pretrained(PATHS.model_path, config=config)
            model.resize_token_embeddings(len(tokenizer))

        # TRAIN WITH TRAINER
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        if USE_REGRESSION: compute_metrics = compute_metrics_for_regression
        else: compute_metrics = compute_metrics_for_classification
        trainer = Trainer( 
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        if LOAD_FROM is None:
                trainer.train()

        # PLOT CONFUSION MATRIX
        y_true = valid['score'].values
        predictions0 = trainer.predict(tokenized_valid).predictions
        if USE_REGRESSION: 
            predictions = predictions0.round(0) + 1
        else: 
            predictions = predictions0.argmax(axis=1) + 1 
        cm = confusion_matrix(y_true, predictions, labels=[x for x in range(1,7)])
        draw_cm = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=[x for x in range(1,7)])
        draw_cm.plot()
        plt.show()

        # SAVE FOLD MODEL AND TOKENIZER
        if LOAD_FROM is None:
            trainer.save_model(f'/ai/users/bst/competition/outputs/{PATHS.model_name}_AES2_fold_{fold}_v{VER}')
            tokenizer.save_pretrained(f'/ai/users/bst/competition/outputs/{PATHS.model_name}_AES2_fold_{fold}_v{VER}')

        # SAVE OOF PREDICTIONS
        if USE_REGRESSION: 
            valid['pred'] = predictions0 + 1 
        else:
            COLS = [f'p{x}' for x in range(CFG.num_labels)] 
            valid[COLS] = predictions0 
        valid.to_csv(f'outputs/valid_df_fold_{fold}_v{VER}.csv', index=False)


if COMPUTE_CV:
    dfs = []
    for k in range(CFG.n_splits):
        dfs.append( pd.read_csv(f'outputs/valid_df_fold_{k}_v{VER}.csv') )
        os.system(f'rm outputs/valid_df_fold_{k}_v{VER}.csv')
    dfs = pd.concat(dfs)
    dfs.to_csv(f'outputs/valid_df_v{VER}.csv',index=False)
    print('Valid OOF shape:', dfs.shape )

if COMPUTE_CV:
    if USE_REGRESSION:
        m = cohen_kappa_score(dfs.score.values, dfs.pred.values.clip(1,6).round(0), weights='quadratic')
    else:
        m = cohen_kappa_score(dfs.score.values, dfs.iloc[:,-6:].values.argmax(axis=1)+1, weights='quadratic')
    print('Overall QWK CV =',m)

test = pd.read_csv(PATHS.test_path)
print('Test shape:', test.shape )

all_pred = []
test['label'] = 0.0

for fold in range(CFG.n_splits):
    
    # LOAD TOKENIZER
    if LOAD_FROM:
        tokenizer = AutoTokenizer.from_pretrained( f'/ai/users/bst/competition/outputs/{PATHS.model_name}_AES2_fold_{fold}_v{VER}')
    else:
        tokenizer = AutoTokenizer.from_pretrained(f'/ai/users/bst/competition/outputs/{PATHS.model_name}_AES2_fold_{fold}_v{VER}')
    tokenize = Tokenize(test, test, tokenizer)
    tokenized_test, _, _ = tokenize()

    # LOAD MODEL
    if LOAD_FROM:
        model = AutoModelForSequenceClassification.from_pretrained( f'/ai/users/bst/competition/outputs/{PATHS.model_name}_AES2_fold_{fold}_v{VER}')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(f'/ai/users/bst/competition/outputs/{PATHS.model_name}_AES2_fold_{fold}_v{VER}')
    
    # INFER WITH TRAINER
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=tokenized_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # SAVE PREDICTIONS
    predictions = trainer.predict(tokenized_test).predictions
    all_pred.append( predictions )

preds = np.mean(all_pred, axis=0)
print('Predictions shape:',preds.shape)

sub = pd.read_csv(PATHS.submission_path)
if USE_REGRESSION: 
    sub["score"] = preds.clip(0,5).round(0)+1
else: 
    sub["score"] = preds.argmax(axis=1)+1
sub.score = sub.score.astype('int32')
sub.to_csv('submission.csv',index=False)
print('Submission shape:', sub.shape )