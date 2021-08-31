from load_config import config

import pandas as pd
from text_processing import TextProcessing
from compute_missing_attributes import ComputeMissingAttributes
from compute_ampersand_category import ComputeAmpersandCategory

class CatalogIndexer():

    def __init__(self, config, domain, catalog):
        self.config = config
        self.domain = domain
        self.catalog = catalog
        self._validate_catalog()

    def _validate_catalog(self):
        if not type(self.catalog) == pd.core.frame.DataFrame:
            raise Exception('catalog should be a pandas dataframe')
        if not all([input_column in self.catalog.column for input_column in config[self.domain]['input_columns']]):
            raise Exception('input columns mentioned in the config are not present in the given catalog')

    def index_catalog(self):
        
        if self.config[self.domain]['run_text_processor']:
            self.text_processor = TextProcessing()
            self.run_text_processor()

        if self.config[self.domain]['run_ampersand_category']:
            self.compute_ampersand_category = ComputeAmpersandCategory()
            self.run_ampersand_category()

        if self.config[self.domain]['run_missing_attribute']:
            self.compute_missing_attribute = ComputeMissingAttributes()
            self.run_missing_attribute()

        if self.config[self.domain]['run_merge_attributes']:
            pass
            

    def run_text_processor(self, catalog):
        pass

    def run_ampersand_category(self, catalog):
        pass

    def run_missing_attribute(self, catalog):
        self.catalog = self.compute_missing_attribute.compute(self.catalog, self.config[self.domain])