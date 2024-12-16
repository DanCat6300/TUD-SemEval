# %%
import os
from sys import argv
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import DistilBertTokenizer, TrainingArguments, Trainer, DistilBertForSequenceClassification
import torch
from torch.nn.functional import softmax
from sklearn.metrics import f1_score, accuracy_score

parser = ArgumentParser()
parser.add_argument("--output_directory", type=str)
parser.add_argument("--language", type=str)
parser.add_argument("--context_length", type=int, default=128)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--cache_dir", type=str, default=".")
args, _ = parser.parse_known_args(argv[1:])
DIRECTORY = args.output_directory

# os.makedirs(f"{DIRECTORY}/distilbert_proj/tokenizer", exist_ok=True)

# # Check which component is used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
dataset = load_dataset("csv", data_files="eng.csv")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocess function
def prepare_data(rows):
    labels = list(zip(
        rows["Anger"], 
        rows["Fear"], 
        rows["Joy"], 
        rows["Sadness"], 
        rows["Surprise"]
    ))
    labels = [[float(label) for label in example] for example in labels]
    tokenized_data = tokenizer(rows["text"], truncation=True, padding="max_length")
    tokenized_data["labels"] = labels
    return tokenized_data.to(device)

# Apply preprocessing
tokenized_dataset = dataset.map(prepare_data, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["id", "text", "Anger", "Fear", "Joy", "Sadness", "Surprise"])


# %%
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]


# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0).astype(int)
    f1 = f1_score(labels, predictions, average="micro")
    accuracy = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": accuracy}


# %%
base_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=5, 
    problem_type="multi_label_classification"
).to(device)

base_training_args = TrainingArguments(
    output_dir="./base_output",
    num_train_epochs=3,
    logging_dir="./logs_base_model",
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    fp16=True
)

base_trainer = Trainer(
    model=base_model,
    args=base_training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
base_trainer.train()
base_result = base_trainer.evaluate()
print("Base model results:", base_result)
base_trainer.save_model(f"{DIRECTORY}/distilbert_proj")

# %%
base_model.eval()
text = "Tomorrow is my birthday"
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

with torch.no_grad():
    outputs = base_model(**inputs)

logits = outputs.logits

probability = softmax(logits, dim=1)

predicted_class = torch.argmax(probability, dim=1).item()
print(f"Predicted emotion: {predicted_class}, Probabilities: {probability}")
