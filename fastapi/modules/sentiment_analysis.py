import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AutoTokenizer


class SentimentAnalysis:

    def __init__(self):
        model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)


    @staticmethod
    def preprocess(text):
        """
        Preprocess text (username and link placeholders)
        """
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def predict(self, text):
        text = SentimentAnalysis.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        l_scores = {}
        for i in range(scores.shape[0]):
            l = self.config.id2label[ranking[i]]
            s = scores[ranking[i]]
            l_scores[l] = np.round(float(s), 4)

        return l_scores