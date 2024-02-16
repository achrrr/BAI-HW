import random

from PIL import Image
from transformers import pipeline

pipelines = {}

classifier = pipeline("sentiment-analysis")
res = classifier("My dog does not listen")
pipelines["sentiment-analysis"] = res

generator = pipeline("text-generation", model="facebook/opt-2.7b")
res = generator(
    "Today we will",
    max_length=random.randint(20, 50),
    num_return_sequences=1
)
pipelines["text-generation"] = res

classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
res = classifier(Image.open("images/titanic-dicaprio-winslet.jpg"))
pipelines["image-classification"] = res

for pline, result in pipelines.items():
    print(f"result from pipeline {pline}: {result}")
