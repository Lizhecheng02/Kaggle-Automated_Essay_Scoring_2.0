## This Repo is for [Kaggle - Automated Essay Scoring 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2)

### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```

### Prepare Data

#### 1. Set Kaggle Api

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
```

#### 2. Install unzip

```bash
sudo apt install unzip
```

#### 2. Download Dataset
```bash
cd dataset
kaggle competitions download -c learning-agency-lab-automated-essay-scoring-2
unzip learning-agency-lab-automated-essay-scoring-2.zip
kaggle datasets download -d lizhecheng/aes2-0-train-dataset
unzip aes2-0-train-dataset.zip
```

### Run Deberta Regression

#### 1. If you want to run with specific pooling method (no awp)
```bash
cd deberta
cd xxxPooling
chmod +x ./regression.sh
./regression.sh
```
#### 2. If you want to run with specific pooling method (awp)
```bash
cd deberta-awp
cd xxxPooling
chmod +x ./regression.sh
./regression.sh
```

#### 3. If you want the flexibility to set the parameters of the model
```bash
cd src
(change the settings in config.py)
python train.py
```

### Run LLM

#### 1. Classification
```bash
cd llm
chmod +x ./classification.sh
./classification.sh
```

#### 2. Regression
```bash
cd llm
chmod +x ./regression.sh
./regression.sh
```

### Run Tree Models
```bash
cd tree
(change the settings in config.py)
python full_cv_main.py / python out_of_fold_cv_main.py
```

### Some Tricks (Replace "\n\n" ...)
```bash
cd replace
chmod +x ./regression.sh
./regression.sh
```
You can also combine tricks into the codes under **src** directory 


### Conclusion 
#### 1. It is very important to use a tree model as a two-stage correction in this competition.
#### 2. You don't need to submit your entry for the Learning Agency Lab competition in the final month.
#### 3. A meaningless competition, a meaningless bronze medal.

