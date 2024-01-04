from transformers import BertTokenizer, BertForSequenceClassification, pipeline

class NewsClassifier():
    def __init__(self, model="yiyanghkust/finbert-tone"):
        self.finbert   = BertForSequenceClassification.from_pretrained(model, num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained(model)

    @property
    def nlp(self):
        return pipeline("text-classification", model=self.finbert, tokenizer=self.tokenizer)

    def sentiment_for(self, title):
        return self.nlp(title)[0]