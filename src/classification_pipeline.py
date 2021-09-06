import pandas as pd

# data transformation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

# binary relevance
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

# classification models
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# performance metric
from sklearn.metrics import f1_score

# model pipeline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer 

# text processing library
# from text_processing import TextProcessing

# class TextProcessor(BaseEstimator):

#     def __init__(self, text_preprocessing_model, text_column):
#         self.text_column = text_column
#         self.text_preprocessing_model = text_preprocessing_model

#     def fit(self, documents, y=None):
#         return self

#     def transform(self, x_dataset):
#         x_dataset['cleaned_text'] = x_dataset[self.text_column].apply(lambda x: self.text_preprocessing_model.clean_text(x))
#         return x_dataset
# import sister

class SentenceEmbedding():

    def __init__(self, embedding_type = None):
        if embedding_type == 'bert':
            self.sentence_embedding = sister.BertEmbedding(lang="en")
        else:
            self.sentence_embedding = sister.MeanEmbedding(lang="en")
        # if embedding_type == 'bert':
            # self.sentence_embedding = sentence_embedding
        # else:
            # self.sentence_embedding = sentence_embedding_bert

    def get_sentence_embedding(self, sentence):
        if type(sentence) == str:
            return self.sentence_embedding(sentence)
        else:
            return [self.sentence_embedding(str(sent)) for sent in list(sentence)]

class TextVectorizer(BaseEstimator):

    def __init__(self, text_column, vectorizer_algorithm, embedding_type = None):
        self.text_column = text_column
        self.vectorizer_algorithm = vectorizer_algorithm
        self.vectorizer = None
        self.sentence_embedding = SentenceEmbedding(embedding_type) if vectorizer_algorithm == 'sentence_embeddings' else None

    def fit(self, x_dataset, y=None):
        if self.vectorizer_algorithm == 'count_vectorizer':
            pass
        elif self.vectorizer_algorithm == 'tfidf_vectorizer':
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(x_dataset)
        elif self.vectorizer_algorithm == 'sentence_embeddings':
            pass
        else:
            raise Exception(f'invalid vectorizer_algorithm: {vectorizer_algorithm}')
        return self

    def transform(self, x_dataset):
        if self.vectorizer_algorithm == 'count_vectorizer':
            pass
        elif self.vectorizer_algorithm == 'tfidf_vectorizer':
            x_dataset = self.vectorizer.transform(x_dataset)
        elif self.vectorizer_algorithm == 'sentence_embeddings':
            self.sentence_embedding.get_sentence_embedding(x_dataset)
        else:
            raise Exception(f'invalid vectorizer_algorithm: {vectorizer_algorithm}')
        return x_dataset

class TextClassification():

    def __init__(self, data, text_column, label_column, model_config):
        
        # assigning params to instance variable
        self.data = data
        self.text_column = text_column
        self.label_column = label_column
        self.classification_type = model_config['classification_type']
        self.model_algorithm = model_config['model_algorithm']
        self.vectorizer_algorithm = model_config['vectorizer_algorithm']
        
        # self.text_preprocessing_model = text_preprocessing_model

        # process data
        self.process_data()

        #building pipeline
        self._build_model_pipeline()

    def process_data(self,):
        
        self.data = self.data[[self.text_column, self.label_column]]
        self.data.dropna(inplace = True)
        
        # transforming target label
        if self.classification_type == 'multi-class':
            self.encoder = LabelEncoder()
            self.data[self.label_column] = self.encoder.fit_transform(self.data[self.label_column])
        else:
            self.encoder = MultiLabelBinarizer()
            self.data[self.label_column] = self.encoder.fit_transform(self.data[self.label_column])

        # splitting of training and test data
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.data[self.text_column], self.data[self.label_column], test_size=0.2, shuffle=True, random_state=1)

    # building end to end pipeline
    def _build_model_pipeline(self,):

        pipeline_steps = list()
        pipeline_steps.append(('text_vectorizer', TextVectorizer(self.text_column, self.vectorizer_algorithm)))
        # pipeline_steps.append(('column_transformation', self._load_column_transformer()))

        if self.model_algorithm == 'LR' or self.model_algorithm == None:
            pipeline_steps.append(('LogisticRegression', OneVsRestClassifier(LogisticRegression())))

        elif self.model_algorithm == 'NB':
            pipeline_steps.append(('MultinomialNB', OneVsRestClassifier(MultinomialNB())))

        elif self.model_algorithm == 'SVC':
            pipeline_steps.append(('LinearSVC', OneVsRestClassifier(LinearSVC())))

        elif self.model_algorithm == 'XGB':
            pipeline_steps.append(('XGBClassifier', OneVsRestClassifier(XGBClassifier(verbosity = 0))))

        self.model_pipeline = Pipeline(steps=pipeline_steps)

    # pipeline component for transforming any columns in input dataframe
    def _load_column_transformer(self,):
        return None

    # training pipeline
    def train_pipeline(self,):
        self.model_pipeline.fit(self.train_x, self.train_y)

    # evaluating pipeline
    def evaluate_pipeline(self,):
        y_pred = self.model_pipeline.predict(self.test_x)
        metric_score = f1_score(self.test_y, y_pred, average="micro")
        print(f'F1 score for {self.model_algorithm}: {metric_score}')

    # prediction using trained pipeline
    def predict(self, input_text):
        input_text = [input_text]
        prediction = self.model_pipeline.predict(input_text)
        return self.encoder.inverse_transform(prediction)
