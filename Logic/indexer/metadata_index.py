from .index_reader import Index_reader
from .indexes_enum import Indexes, Index_types
import json

class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        self.path = path
        self.documents = self.read_documents(path)
        self.metadata_index = self.create_metadata_index()
        self.store_metadata_index(path)

    def read_documents(self, path):
        """
        Reads the documents.
        
        """
        documents_reader = Index_reader(path, index_name=Indexes.DOCUMENTS)
        return documents_reader.index


    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['averge_document_length'] = {
            'characters': self.get_average_document_field_length('characters'),
            'genres': self.get_average_document_field_length('genres'),
            'descriptions': self.get_average_document_field_length('description')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    def get_average_document_field_length(self,where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        ans = 0
        total_length = 0
        doc_count = 0
        
        for doc_id, doc in self.documents.items():
            if where == 'characters':
                field = doc.get('characters', [])
            elif where == 'genres':
                field = doc.get('genres', [])
            elif where == 'description':
                field = doc.get('description', '')
            else:
                continue
            
            if isinstance(field, list):
                length = len(field)
            elif isinstance(field, str):
                length = len(field.split())
            else:
                length = 0
            
            total_length += length
            doc_count += 1
        
        if doc_count > 0:
            ans = total_length / doc_count
        
        return ans

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path =  path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


    
if __name__ == "__main__":
    meta_index = Metadata_index('index/')