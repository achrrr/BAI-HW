import torch
import torch.nn.functional as F

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")
res = classifier("I've been waiting for a HuggingFace course my whole life")

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
res = classifier("I've been waiting for a HuggingFace course my whole life")
print(res)

seq = "Using a Transfomer network is simple"
res = tokenizer(seq)
print(f"token: {res}")
tokens = tokenizer.tokenize(seq)
print(f"tokens: {tokens}")
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"ids: {ids}")
decoded_string = tokenizer.decode(ids)
print(decoded_string)


model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
X_train = [
    "I've been waiting for a HuggingFace course my whole life",
    "Python is great!",
    "AI is innovative!"
]
res= classifier(X_train)
print(res)
 
batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)

