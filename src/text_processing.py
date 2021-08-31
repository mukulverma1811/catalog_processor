import inflect, spacy, json
from collections import defaultdict

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

    def __init__(self,):
        self.inflect_engine = inflect.engine()
        self.spacy_nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])

    def _get_singular_token(self, token):
        processed_token = self.inflect_engine.singular_noun(token)
        if processed_token:
            return processed_token
        return token

    def clean_text(self, text):
        return self.clean_texts([text])[0]

    def clean_texts(self, texts):
        processed_texts = list()
        docs = list(self.spacy_nlp.pipe(texts))
        for doc in docs:
            original_text = " ".join([token.text for token in doc])
            processed_text = " ".join([self._get_singular_token(token.lemma_) if token.lemma_ != '-PRON-' else token.lower_ for token in doc if (not token.is_punct or token.lemma_ == "%" ) and not token.is_space])
            processed_text = processed_text if len(processed_text) > 1 else original_text
            processed_texts.append(processed_text.lower())
        return processed_texts

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