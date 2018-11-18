import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class BaseLines:

    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.validation_df = None
        self.y_test = None
        self.vectorizer = TfidfVectorizer()

    def laodData(self):
        # Load Data
        print("Loading data")
        self.train_df = pd.read_csv("C:/Users/anubh/Documents/NLP/Project/data/train.csv")
        print("Read train_df")
        self.test_df = pd.read_csv("C:/Users/anubh/Documents/NLP/Project/data/test.csv")
        print("Read test")
        self.validation_df = pd.read_csv("C:/Users/anubh/Documents/NLP/Project/data/valid.csv")
        print("Read valid")
        self.y_test = np.zeros(len(self.test_df))
        print("Initialized y_test")

    def evaluate_recall(self, y, y_test, k=1):
        num_examples = float(len(y))
        num_correct = 0
        for predictions, label in zip(y, y_test):
            if label in predictions[:k]:
                num_correct += 1
        return num_correct / num_examples

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values,data.Utterance.values))

    def predictTFIDF(self, context, utterances):
        print("predicting TFIDF")
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]

    def predict_random(self, context, utterances):
        print("predicting random")
        return np.random.choice(len(utterances), 10, replace=False)


    def evaluateRandomPredictor(self):
        print("Evaluating Random Predictor")
        # Evaluate Random predictor
        y_random = [self.predict_random(self.test_df.Context[x], self.test_df.iloc[x, 1:].values) for x in range(len(self.test_df))]
        for n in [1, 2, 5, 10]:
            print("Recall @ ({}, 10): {:g}".format(n, self.evaluate_recall(y_random, self.y_test, n)))

    def evaluateTFIDFPredictor(self):
        print("Evaluating TFIDF")
        self.train(self.train_df)
        print("Trained for TFIDF")
        y = [self.predictTFIDF(self.test_df.Context[x], self.test_df.iloc[x, 1:].values) for x in range(len(self.test_df))]
        for n in [1, 2, 5, 10]:
            print("Recall @ ({}, 10): {:g}".format(n, self.evaluate_recall(y, self.y_test, n)))


def main():
    baseline = BaseLines()
    baseline.laodData()
    baseline.evaluateRandomPredictor()
    baseline.evaluateTFIDFPredictor()


if __name__ == "__main__":
    main()