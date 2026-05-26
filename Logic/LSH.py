import numpy as np
import itertools
import random
import json


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        self.shingled_docs = []
        self.shingle_to_idx = {}
        self.all_shingles = set()

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        words = document.split()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        self.shingled_docs = []
        self.all_shingles = set()
        
        for doc in self.documents:
            shingles = self.shingle_document(doc, 2)
            self.shingled_docs.append(shingles)
            self.all_shingles.update(shingles)
        
        self.shingle_to_idx = {shingle: idx for idx, shingle in enumerate(self.all_shingles)}
        
        matrix = np.zeros((len(self.all_shingles), len(self.documents)), dtype=bool)
        
        for doc_idx, shingles in enumerate(self.shingled_docs):
            for shingle in shingles:
                matrix[self.shingle_to_idx[shingle], doc_idx] = 1
        
        return matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        matrix = self.build_characteristic_matrix()
        num_rows, num_cols = matrix.shape
        
        signatures = np.full((self.num_hashes, num_cols), np.inf)
        
        hash_funcs = []
        for i in range(self.num_hashes):
            a = np.random.randint(1, 100000)
            b = np.random.randint(0, 100000)
            c = 1000003
            hash_funcs.append(lambda x, a=a, b=b, c=c: (a * x + b) % c)
        
        for row_idx in range(num_rows):
            row_hash = hash(row_idx)
            for hash_idx, hash_func in enumerate(hash_funcs):
                hash_val = hash_func(row_hash)
                for col_idx in range(num_cols):
                    if matrix[row_idx, col_idx]:
                        if hash_val < signatures[hash_idx, col_idx]:
                            signatures[hash_idx, col_idx] = hash_val
        
        return signatures

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_docs = signature.shape[1]
        buckets = {}
        
        for band in range(bands):
            start_row = band * rows_per_band
            end_row = start_row + rows_per_band
            
            band_signature = signature[start_row:end_row, :]
            
            for doc_idx in range(num_docs):
                band_tuple = tuple(band_signature[:, doc_idx].flatten())
                
                bucket_key = (band, hash(band_tuple))
                
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                
                if doc_idx not in buckets[bucket_key]:
                    buckets[bucket_key].append(doc_idx)
        
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_bands = 25
        signature = self.min_hash_signature()
        ans = self.lsh_buckets(signature, num_bands, self.num_hashes//num_bands)
        return ans

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        if union == 0:
            return 0
        return intersection / union

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)



def main():
    with open('LSHFakeData.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = [item['descriptions'][0] for item in data]
    
    lsh = MinHashLSH(documents, 100)
    
    buckets = lsh.perform_lsh()
    
    lsh.jaccard_similarity_test(buckets, documents)


    
if __name__ == '__main__':
    main()
