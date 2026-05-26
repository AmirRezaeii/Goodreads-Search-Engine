import re
import string
import json
import csv
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# nltk.download('stopwords', quiet=True)

class Preprocessor:
    def __init__(self, custom_stopwords_path='./stopwords.txt'):
        """
        Initialize the preprocessor, compile patterns, load components, etc.
        """
        pattern = r'\S*http\S*|\S*www\S*|\S+\.ir\S*|\S+\.com\S*|\S+\.org\S*|\S*@\S*'
        self.url_pattern = re.compile(pattern, re.IGNORECASE)
        self.punctuation_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        self.stemmer = PorterStemmer()
        
        try:
            with open(custom_stopwords_path, 'r', encoding='utf-8') as f:
                self.custom_stopwords = set(f.read().splitlines())
        except FileNotFoundError:
            self.custom_stopwords = set()
        
        self.nltk_stopwords = set(stopwords.words('english'))
        self.all_stopwords = self.nltk_stopwords.union(self.custom_stopwords)



    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing pipeline to a single text document.
        """
        if not isinstance(text, str):
            text = str(text)
        
        text = self.url_pattern.sub('', text)
        
        text = text.lower()
        
        text = self.punctuation_pattern.sub('', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        
        filtered_words = [word for word in words if word not in self.all_stopwords]
        
        normalized_words = [self.normalize(word) for word in filtered_words]
        
        return ' '.join(normalized_words)

    def remove_stopwords(self, text: str) -> list:
        """
        Remove stopwords from the text.
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.all_stopwords]
        return filtered_words
    
    def normalize(self, word: str) -> str:
        """
        Normalize the text by stemming, lemmatization, etc.

        Parameters
        ----------
        word : str
            The word to be normalized.

        Returns
        ----------
        list
            The normalized word.
        """
        return self.stemmer.stem(word)

    def preprocess_many(self, documents: list) -> list:
        """
        Apply preprocessing pipeline to a list of documents.
        """
        processed_docs = []
        for doc in documents:
            if isinstance(doc, dict):
                processed_doc = {}
                for key, value in doc.items():
                    if isinstance(value, str):
                        processed_doc[key] = self.preprocess_text(value)
                    elif isinstance(value, list):
                        processed_doc[key] = [self.preprocess_text(item) if isinstance(item, str) else item for item in value]
                    else:
                        processed_doc[key] = value
                processed_docs.append(processed_doc)
            elif isinstance(doc, str):
                processed_docs.append(self.preprocess_text(doc))
            else:
                processed_docs.append(doc)
        return processed_docs
    


def preprocess_docs(docs: list):
    """
    Apply preprocessing to specific fields in a list of documents in-place.
    
    Args:
        docs (list): List of document dictionaries to preprocess
        
    Returns:
        None: Modifies the input list in-place
    
    Notes:
        Preprocesses the following fields: title, description, author
        Handles both string and list field types
    """
    preprocessor = Preprocessor()
    
    for doc in docs:
        if 'title' in doc and isinstance(doc['title'], str):
            doc['title'] = preprocessor.preprocess_text(doc['title'])
        
        if 'description' in doc and isinstance(doc['description'], str):
            doc['description'] = preprocessor.preprocess_text(doc['description'])
        
        if 'author' in doc and isinstance(doc['author'], str):
            doc['author'] = preprocessor.preprocess_text(doc['author'])


def csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to JSON format with specific field mapping.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        json_file_path (str): Path where the output JSON file will be saved
        
    Returns:
        None: Writes output directly to JSON file
    
    Notes:
        Maps CSV fields to JSON structure including:
        - id (from bookId)
        - title, author, description
        - genres, characters, languages (split by commas)
        - publish_date, num_pages, avg_rating
    """
    books = []
    
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            book = {
                'id': row.get('bookId', ''),
                'title': row.get('title', ''),
                'author': row.get('author', ''),
                'description': row.get('description', ''),
                'genres': [g.strip() for g in row.get('genres', '').split(',') if g.strip()],
                'characters': [c.strip() for c in row.get('characters', '').split(',') if c.strip()],
                'languages': [l.strip() for l in row.get('languages', '').split(',') if l.strip()],
                'publish_date': row.get('publish_date', ''),
                'num_pages': row.get('num_pages', ''),
                'avg_rating': row.get('avg_rating', '')
            }
            books.append(book)
    
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(books, json_file, indent=2, ensure_ascii=False)


if __name__ == '__main__':


    csv_to_json('top_3000_rated_books.csv','crawled.json')

    
    json_file_path = 'crawled.json'
    with open(json_file_path, "r", encoding='utf-8') as file:
        docs = json.load(file)

    preprocess_docs(docs)

    with open('preprocessed.json', "w") as file:
        file.write(json.dumps(docs))
