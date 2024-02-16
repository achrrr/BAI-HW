from transformers import pipeline

pipelines = {}

classifier = pipeline("sentiment-analysis")
res = classifier("I've been waiting for a HuggingFace course my whole life")
print(f"sentiment-analysis: {res}")

generator = pipeline("text-generation", model="distilgpt2")
res = generator(
    "In this course, we will teach you to",
    max_length=30,
    num_return_sequences=2
)
print(res)

classifier = pipeline("zero-shot-classification")
res = classifier(
    "This is a course about Python list comprehesion",
    candidate_labels=["education", "politics", "business"]
)
print(res)

