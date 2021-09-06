from load_config import config

# import swifter
import pandas as pd
from text_processing import TextProcessing
from compute_missing_attributes import ComputeMissingAttributes
from compute_ampersand_category import ComputeAmpersandCategory

class CatalogIndexer():

    def __init__(self, domain):
        self.domain_config = config[domain]
        self.catalog = pd.read_csv(self.domain_config['raw_catalog_path'])
        # print(self.catalog.shape)
        # print(self.catalog.isnull().sum())
        # self.catalog = self.catalog.sample(10)
        self._validate_catalog()

    def _validate_catalog(self):
        if not type(self.catalog) == pd.core.frame.DataFrame:
            raise Exception('catalog should be a pandas dataframe')
        # if not all([input_column in self.catalog.column for input_column in self.domain_config['input_columns']]):
        #     raise Exception('input columns mentioned in the config are not present in the given catalog')

    def index_catalog(self):
        
        if self.domain_config['run_text_processor']:
            print('\nStage: running text processing ====>')
            self.text_processor = TextProcessing(True, True, False)
            self.run_text_processor()

        if self.domain_config['run_ampersand_category']:
            print('\nStage: fixing ampersand category issue ====>')
            self.compute_ampersand_category = ComputeAmpersandCategory()
            self.run_ampersand_category()

        if self.domain_config['run_missing_attribute']:
            print('\nStage: computing missing attributes ====>')
            self.compute_missing_attribute = ComputeMissingAttributes()
            self.run_missing_attribute()

        if self.domain_config['run_merge_attributes']:
            print('\nStage: merging attributes ====>')
            pass
            
        self.catalog.to_csv(self.domain_config['processed_catalog_filepath'], index = False)

    def run_text_processor(self):
        for column in self.domain_config['columns_for_text_processor']:
            self.catalog[f'{column}'] = self.catalog[column].apply(lambda x: self.text_processor.clean_text(x))
            # self.catalog[f'{column}'] = self.catalog[column].swifter.apply(lambda x: self.text_processor.clean_text(x))
        self.catalog['input_text'] = self.catalog[self.domain_config['input_text_column']].apply(lambda x: self.text_processor.clean_text(x))
        # self.catalog['input_text'] = self.catalog[self.domain_config['input_text_column']].swifter.apply(lambda x: self.text_processor.clean_text(x))
        

    def run_ampersand_category(self):
        pass

    def run_missing_attribute(self):
        self.catalog = self.compute_missing_attribute.compute(self.catalog, self.domain_config)


if __name__ == "__main__":
    ci = CatalogIndexer('jiomart_autosuggest_v2')
    ci.index_catalog()