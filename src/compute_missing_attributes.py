from classification_pipeline import TextClassification

class ComputeMissingAttributes():

    def __init__(self):
        pass
        # self.text_classification = TextClassification()

    def compute(self, catalog, domain_config):
        input_column = domain_config['input_column']
        output_column = domain_config['output_column']

        # text_classification = TextClassification()
        # for output_column in domain_config['columns_for_computation']:
            # model = TextClassification('')


