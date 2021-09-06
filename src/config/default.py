config = {
  'jiomart_autosuggest': {
    'run_text_processor': True,
    'run_ampersand_category': False,
    'run_missing_attribute': True,
    'run_merge_attributes': False,
    'columns_for_text_processor': [],
    'columns_for_ampersand_category': [],
    'columns_for_missing_attributes': ['New_level1','New_level2'],
    # 'columns_for_missing_attributes': ['New_level1','New_level2','New_level3','Valued_level2'],
    'input_text_column': 'New Product Name',
    'raw_catalog_path': '/Users/mukul4.verma/Documents/workspace/catalog_indexer/data/jiomart/raw/catalog_sample.csv',
    'processed_catalog_filepath': '/Users/mukul4.verma/Documents/workspace/catalog_indexer/data/jiomart/processed/catalog_sample.csv',
    'model_config':{
      'classification_type': 'multi-class',
      'model_algorithm': 'SVC',
      'vectorizer_algorithm': 'tfidf_vectorizer'
    }
  },
  'jiomart_autosuggest_v2': {
    'run_text_processor': True,
    'run_ampersand_category': False,
    'run_missing_attribute': True,
    'run_merge_attributes': False,
    'columns_for_text_processor': [],
    'columns_for_ampersand_category': [],
    # 'columns_for_missing_attributes': ['New_level1','New_level2'],
    'columns_for_missing_attributes': ['New_level1','New_level2','New_level3','Valued_level2'],
    'input_text_column': 'New Product Name',
    'raw_catalog_path': '/home/jioapp/catalog_processor/data/jiomart/raw/catalog.csv',
    'processed_catalog_filepath': '/home/jioapp/catalog_processor/data/jiomart/processed/catalog_processed.csv',
    'model_config':{
      'classification_type': 'multi-class',
      'model_algorithm': 'SVC',
      'vectorizer_algorithm': 'tfidf_vectorizer'
    }
  }
}