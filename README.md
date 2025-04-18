<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/0ed66e2a-4b4a-45a9-802b-a7d2a1db8a6c">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/0ed66e2a-4b4a-45a9-802b-a7d2a1db8a6c">
    <img alt="PORTOS logo" src="https://github.com/user-attachments/assets/0ed66e2a-4b4a-45a9-802b-a7d2a1db8a6c" width="30%">
  </picture>
</div

<div align="center">

# FSA_Chatbot_Intent-Recognition
</div>

Intelligent Chatbot for order tracking and order returns with intent recognition using DistillBERT which is a lighter and faster than its State of the Art Parent Model BERT

Before we Begin this project is a work in progress I will keep updating its features from time to time.

First Lets setup the environment
### Using Python Venv
```
python -m venv portos_env
```
To activate the environment use
```
portos_env\Scripts\activate
```
if you are using anaconda to setup your virtual environment
### Conda Env
```
conda create -n portos_env
```
if you need a specific python version
```
conda create -n portos_env python=3.10
```
And to activate the environment
```
conda activate portos_env
```
Once your done creating your environment(same for conda or venv) use
```
pip install requirements.txt
```

---

## Predicting Intent
Model types: logreg, svm, lstm, bert
If you want to check the intent of a Text you can run this on your CLI of your activated environment
```
python predict_intent.py --model <Model_type> --text <TEXT>
```
Example:
lets say, I want to use the logistic regression to predict the intent and my text is "I want to track my order"
```
python predict_intent.py --model logreg --text "I want to track my order"
```

---

## Running the Chatbot
<div align="center">
  <img alt="PORTOS logo" src="https://github.com/user-attachments/assets/167d3501-770e-46ac-85cb-5e34f24b2925" width="100%">
</div>

### Run THe chatbot
```
python -W ignore::FutureWarning chatbot_with_bert.py
```
the warning tags just avoid any warning showed and help keep the CLI clean
