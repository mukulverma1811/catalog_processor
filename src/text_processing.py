import inflect, spacy, json
from functools import lru_cache
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = set(stopwords.words('english'))

# from spell import SpellCheck

def write_file(texts, file_path):
    texts = [str(text) for text in texts]
    with open(file_path, "w") as outfile:
        outfile.write("\n".join(texts)) 

def write_json_file(data, file_path):
    with open(file_path, "w") as outfile:
        json.dump(data, outfile)

def read_file(file_path) :
    with open(file_path) as f:
        lines = [line.rstrip() for line in f]
        return lines

class TextProcessing:

    def __init__(self, remove_stopword, remove_alpha_numeric, perform_spell_check):
        
        self.remove_stopword = remove_stopword
        self.remove_alpha_numeric = remove_alpha_numeric
        self.perform_spell_check = perform_spell_check

        if self.perform_spell_check:
            self.spellcheck_obj = SpellCheck()

        self.inflect_engine = inflect.engine()
        self.spacy_nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])


    def _get_singular_token(self, token):
        #token's singular noun extracted, if it exists it is returned
        processed_token = self.inflect_engine.singular_noun(token)
        if processed_token:
            return processed_token
        return token
    
    @lru_cache(maxsize=100000)
    def clean_text(self, text):
        text = str(text)
        if text == 'nan':
            return text
        return self.clean_texts([text])[0]

    def clean_texts(self, texts):
        processed_texts = list()
        texts = [str(text) for text in texts]
        docs = list(self.spacy_nlp.pipe(texts))
        for doc in docs:
            
            processed_text = [self._get_singular_token(token.lemma_) if token.lemma_ != '-PRON-' else token.lower_ \
                                       for token in doc if (not token.is_punct or token.lemma_ == "%" ) \
                                       and not token.is_space]
            
            if self.remove_stopword:
                processed_text = [token for token in processed_text if token not in STOPWORDS]
                
            if self.remove_alpha_numeric:
                processed_text = [token for token in processed_text if token.isalpha()]
                
            if self.perform_spell_check:
                processed_text = [self.spellcheck_on_token(token) for token in processed_text]
            
            if len(processed_text) > 1:
                processed_text = " ".join(processed_text)
            else:
                processed_text = " ".join([token.text for token in doc])
            
            processed_texts.append(processed_text.lower())
        return processed_texts
    
    def spellcheck_on_token(self, token):
        return self.spellcheck_obj.generate_candidates(token)
        
    def clean_file(self, filename):
        texts = read_file(filename)
        texts = list(set([text.lower() for text in texts]))
        return self.clean_texts(texts)

    def token_to_texts_mapping(self, texts, clean = False):
        token_to_texts_map = defaultdict(list)
        for text in texts:
            text = self.clean_text(text) if clean else text
            for token in text.split():
                token_to_texts_map[token].append(text)
        for key in token_to_texts_map:
            token_to_texts_map[key] = list(set(token_to_texts_map[key]))
        return token_to_texts_map