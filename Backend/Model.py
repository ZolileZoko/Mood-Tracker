from transformers import pipeline

classifier = pipeline("text-classification", 
                      model="bhadresh-savani/distilbert-base-uncased-emotion")

def classify_emotion(text):
    result = classifier(text)
    return result[0]['label'], result[0]['score']
