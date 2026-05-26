import json
from .indexes_enum import Indexes, Index_types
from .index_reader import Index_reader

class DocumentLengthsIndex:
    def __init__(self, path='index/'):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        self.path = path
        # Load the documents index (contains actual document content)
        self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index
        
        self.document_length_index = {
            Indexes.CHARACTERS: self.get_documents_length(Indexes.CHARACTERS),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES),
            Indexes.DESCRIPTIONS: self.get_documents_length(Indexes.DESCRIPTIONS)
        }
        
        self.store_document_lengths_index(path, Indexes.CHARACTERS)
        self.store_document_lengths_index(path, Indexes.GENRES)
        self.store_document_lengths_index(path, Indexes.DESCRIPTIONS)

    def get_documents_length(self, where):
        """
        Gets the documents' length for the specified field.

        Parameters
        ----------
        where : Indexes
            The field to get the document lengths for.

        Returns
        -------
        dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field (where).
        """
        document_lengths = {}
        
        # Iterate through each document in the documents index
        for doc_id, doc_content in self.documents_index.items():
            # Get the field content based on the field type
            if where == Indexes.CHARACTERS:
                field_content = doc_content.get('characters', [])
            elif where == Indexes.GENRES:
                field_content = doc_content.get('genres', [])
            elif where == Indexes.DESCRIPTIONS:
                field_content = doc_content.get('description', '')
            else:
                field_content = ''
            
            # Calculate length based on content type
            if isinstance(field_content, list):
                length = len(field_content)
            elif isinstance(field_content, str):
                length = len(field_content.split())
            else:
                length = 0
            
            document_lengths[doc_id] = length
        
        return document_lengths

    def store_document_lengths_index(self, path, index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        file_path = f"{path}{index_name.value}_{Index_types.DOCUMENT_LENGTH.value}_index.json"
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.document_length_index[index_name], file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    document_lengths_index = DocumentLengthsIndex('index/')
    print('Document lengths index stored successfully.')