from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# Load an amazon review dataset from huggingface
dataset = load_dataset("amazon_polarity")

def tokenize_function(data):
    return tokenizer(data["content"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(output_dir="test-trainer")
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)
trainer.train()