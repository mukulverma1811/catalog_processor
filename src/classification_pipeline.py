# import sys, os
# sys.path.append('../src')

import pandas as pd

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
from text_processing import TextProcessing

# class TextProcessor(BaseEstimator):

#     def __init__(self, text_preprocessing_model, text_column):
#         self.text_column = text_column
#         self.text_preprocessing_model = text_preprocessing_model

#     def fit(self, documents, y=None):
#         return self

#     def transform(self, x_dataset):
#         x_dataset['cleaned_text'] = x_dataset[self.text_column].apply(lambda x: self.text_preprocessing_model.clean_text(x))
#         return x_dataset

class TextVectorizer(BaseEstimator):

    def __init__(self, text_column, vectorizer_algorithm):
        self.text_column = text_column
        self.vectorizer_algorithm = vectorizer_algorithm

    def fit(self, x_dataset, y=None):
        if self.vectorizer_algorithm == 'count_vectorizer':
            pass
        elif self.vectorizer_algorithm == 'tfidf_vectorizer':
            pass
        elif self.vectorizer_algorithm == 'embeddings_vectorizer':
            pass
        else:
            raise Exception(f'invalid vectorizer_algorithm: {vectorizer_algorithm}')
        return self

    def transform(self, x_dataset):
        if self.vectorizer_algorithm == 'count_vectorizer':
            pass
        elif self.vectorizer_algorithm == 'tfidf_vectorizer':
            pass
        elif self.vectorizer_algorithm == 'embeddings_vectorizer':
            pass
        else:
            raise Exception(f'invalid vectorizer_algorithm: {vectorizer_algorithm}')
        return x_dataset


class TextClassification():

    def __init__(self, raw_data, text_column, label_column, model_algorithm, classification_type, text_preprocessing_model = None):
        
        # assigning params to instance variable
        self.raw_data = raw_data
        self.text_column = text_column
        self.label_column = label_column
        self.model_algorithm = model_algorithm
        self.classification_type = classification_type
        self.text_preprocessing_model = text_preprocessing_model

        # loading data
        self.process_raw_data()

        #building pipeline
        self._build_model_pipeline()

    # def _load_data(self,):
    #     file_extension = self.raw_data_filepath.split(".")[-1]
    #     try:
    #         if file_extension == ".csv":
    #             self.raw_data = pd.read_csv(self.raw_data_filepath)

    #             self.train_x = 
    #             self.train_y = 
    #             self.test_x = 
    #             self.test_y = 
    #         else:
    #             raise ValueError (f'{file_extension} file type is not supported')
    #     except ValueError as ve:
    #         print(f'error occured while loading data: {ve}')
    #     except Exception as e:
    #         print(f'error occured while loading data: {e}')

    def process_raw_data(self,):
        pass

    # building end to end pipeline
    def _build_model_pipeline():

        pipeline_steps = list()
        # pipeline_steps.append(('text_preprocessing', TextProcessor()))
        pipeline_steps.append(('text_vectorizer', TextVectorizer()))
        pipeline_steps.append(('column_transformation', self._load_column_transformer()))

        if self.model_algorithm == 'LR' or self.model_algorithm == None:
            pipeline_steps.append(('LogisticRegression', LogisticRegression()))

        elif self.model_algorithm == 'NB':
            pipeline_steps.append(('MultinomialNB', MultinomialNB()))

        elif self.model_algorithm == 'SVC':
            pipeline_steps.append(('LinearSVC', LinearSVC()))

        elif self.model_algorithm == 'XGB':
            pipeline_steps.append(('XGBClassifier', XGBClassifier()))

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
        return self.model_pipeline.predict(input_text)
