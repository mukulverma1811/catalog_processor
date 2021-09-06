from classification_pipeline import TextClassification

class ComputeMissingAttributes():

    def __init__(self):
        pass
        # self.text_classification = TextClassification()

    def compute(self, catalog, domain_config):

        model_config = domain_config['model_config']

        for column in domain_config['columns_for_missing_attributes']:
            
            model = TextClassification(catalog, 'input_text', column, model_config)
            
            print(f'\ntraining model for computing missing attributes for the column : {column}')
            model.train_pipeline() 
            model.evaluate_pipeline()

            print(f'predicting and saving missing attributes')
            catalog[column] = catalog.swifter.apply(lambda row : self.get_missing_attribute(model, row['input_text'], row[column]), axis = 1)
        
        return catalog

    def get_missing_attribute(self, model, text, label):
        if type(text) == str and (type(label) != str or len(label) < 1 or 'dummy' in label):
            return model.predict(text)[0]
        return label


