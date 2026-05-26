import math
from collections import Counter


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The inverted index with structure {term: {document_id: tf}}.
        number_of_documents : int
            The number of documents in the collection.
        """
        self.index = index
        self.idf = {}
        self.N = max(int(number_of_documents), 1)
        self._collection_frequencies = None
        self._collection_length = None
        self._prepare_collection_stats()

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.
        """
        docs = set()
        query_terms = query.split()
        for term in query_terms:
            if term in self.index:
                docs.update(self.index[term].keys())
        return list(docs)

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.
        """
        if term in self.idf:
            return self.idf[term]
        
        if term not in self.index:
            return 0.0
        
        df = len(self.index[term])
        idf_value = math.log(self.N / (df + 1e-10))
        self.idf[term] = idf_value
        return idf_value

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.
        """
        query_terms = query.split()
        return Counter(query_terms)

    def compute_scores_with_vector_space_model(self, query, method):
        """
        Compute scores with vector space model.
        """
        if method == 'OkapiBM25' or method == 'bm25':
            return {}
        
        scores = {}
        query_tfs = self.get_query_tfs(query)
        documents = self.get_list_of_documents(query)
        
        query_method, document_method = method.split('.')
        
        query_weights = {}
        for term, tf in query_tfs.items():
            applied_tf = self._apply_tf(tf, query_method[0])
            idf_val = self.get_idf(term)
            if query_method[1] == 't':
                applied_tf *= idf_val
            query_weights[term] = applied_tf
        
        if query_method[2] == 'c':
            query_weights = self._cosine_normalize(query_weights)
        
        for doc_id in documents:
            score = self.get_vector_space_model_score(
                query, query_weights, doc_id, document_method, query_method
            )
            if score > 0:
                scores[doc_id] = score
        
        return scores


    def get_vector_space_model_score(
        self, query, query_weights, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.
        """
        score = 0.0
        query_terms = set(query_weights.keys())
        
        doc_weights = {}
        for term in query_terms:
            if term in self.index and document_id in self.index[term]:
                tf = self.index[term][document_id]
                applied_tf = self._apply_tf(tf, document_method[0])
                if document_method[1] == 't':
                    idf_val = self.get_idf(term)
                    applied_tf *= idf_val
                doc_weights[term] = applied_tf
        
        dot_product = 0.0
        for term in query_terms:
            query_weight = query_weights.get(term, 0)
            doc_weight = doc_weights.get(term, 0)
            dot_product += query_weight * doc_weight
        
        query_norm = 1.0
        doc_norm = 1.0
        
        if query_method[2] == 'c':
            query_norm = math.sqrt(sum(w ** 2 for w in query_weights.values()))
            if query_norm > 0:
                query_norm = 1.0
        
        if document_method[2] == 'c':
            full_doc_norm = self._get_document_norm(document_id, document_method)
            if full_doc_norm > 0:
                doc_norm = full_doc_norm
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return dot_product / (query_norm * doc_norm)

    def _get_document_norm(self, document_id, document_method):
        """
        Calculate the cosine norm of a full document vector.
        """
        norm = 0.0
        for term, postings in self.index.items():
            if document_id in postings:
                tf = postings[document_id]
                weight = self._apply_tf(tf, document_method[0])
                if document_method[1] == 't':
                    idf_val = self.get_idf(term)
                    weight *= idf_val
                norm += weight ** 2
        
        return math.sqrt(norm)

    def compute_scores_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        Compute scores with Okapi BM25.
        """
        scores = {}
        query_terms = query.split()
        documents = self.get_list_of_documents(query)
        
        k1 = 1.5
        b = 0.75

        if average_document_field_length == 0:
            average_document_field_length = 1.0
        
        for doc_id in documents:
            score = self.get_okapi_bm25_score(
                query, doc_id, average_document_field_length, document_lengths, k1, b
            )
            if score > 0:
                scores[doc_id] = score
        
        return scores

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths, k1=1.5, b=0.75
    ):
        """
        Returns the Okapi BM25 score of a document for a query.
        """
        score = 0.0
        query_terms = set(query.split())
        doc_length = document_lengths.get(document_id, 0)
        
        if average_document_field_length == 0:
            average_document_field_length = 1.0

        for term in query_terms:
            if term not in self.index or document_id not in self.index[term]:
                continue
            
            tf = self.index[term][document_id]
            idf = self.get_idf(term)
            
            if idf == 0:
                continue
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / average_document_field_length))
            score += idf * (numerator / denominator)
        
        return score

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates scores for each document based on the unigram model.
        """
        scores = {}
        query_terms = query.split()
        documents = self.get_list_of_documents(query)
        
        for doc_id in documents:
            score = self.compute_score_with_unigram_model(
                query, doc_id, smoothing_method, document_lengths, alpha, lamda
            )
            scores[doc_id] = score
        
        total_scores = sum(scores.values())
        if total_scores > 0:
            for doc_id in scores:
                scores[doc_id] = scores[doc_id] / total_scores
        
        return scores

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the unigram score of a document for a query.
        """
        query_terms = query.split()
        doc_length = document_lengths.get(document_id, 0)
        
        if doc_length == 0:
            return 0.0
        
        log_score = 0.0
        
        for term in query_terms:
            tf = self.index.get(term, {}).get(document_id, 0)
            
            collection_freq = self._collection_frequencies.get(term, 0)
            collection_prob = collection_freq / self._collection_length if self._collection_length > 0 else 0
            
            doc_prob = tf / doc_length if doc_length > 0 else 0
            
            if smoothing_method == 'naive':
                prob = doc_prob
            elif smoothing_method == 'bayes':
                prob = (tf + alpha * collection_prob) / (doc_length + alpha)
            elif smoothing_method == 'mixture':
                prob = lamda * doc_prob + (1 - lamda) * collection_prob
            else:
                prob = doc_prob
            
            if prob == 0:
                prob = 1e-10
            
            log_score += math.log(prob)
        
        return math.exp(log_score) if log_score != 0 else 0.0

    def _apply_tf(self, tf, mode):
        """
        Apply term frequency (tf) weighting based on the specified mode.
        mode (str): Weighting scheme:
            - 'n'
            - 'l'

        """
        if mode == 'n':
            return tf
        elif mode == 'l':
            return 1 + math.log(tf) if tf > 0 else 0
        else:
            return tf

    def _cosine_normalize(self, weights):
        """
        Normalize a vector of term weights using cosine normalization.
        """
        norm = math.sqrt(sum(w ** 2 for w in weights.values()))
        if norm == 0:
            return weights
        return {term: w / norm for term, w in weights.items()}

    def _prepare_collection_stats(self):
        """
        Compute and cache collection-wide statistics for the index.
        """
        self._collection_frequencies = {}
        self._collection_length = 0
        
        for term, postings in self.index.items():
            total_tf = sum(postings.values())
            self._collection_frequencies[term] = total_tf
            self._collection_length += total_tf
