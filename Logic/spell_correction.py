import pickle
import os
from collections import Counter

class SpellCorrection:
    def __init__(self, all_documents=None, load_path=None, save_path=None):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str, optional
            The input documents used to build the vocabulary.
        load_path : str, optional
            Path to load precomputed data from.
        save_path : str, optional
            Path to save computed data to.
        """
        if load_path and os.path.exists(load_path):
            self.load(load_path)
        elif all_documents is not None:
            self.all_k_gram_words, self.word_counter = self.k_gramming_and_counting(all_documents)
            if save_path:
                self.save(save_path)
        else:
            self.all_k_gram_words = {}
            self.word_counter = {}

    def k_gram_word(self, word, k=2):
        """
        Convert a word into a set of k-grams.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each k-gram.

        Returns
        -------
        set
            A set of k-grams.
        """
        if len(word) < k:
            return {word}
        
        k_grams = set()
        padded_word = '$' + word + '$'
        
        for i in range(len(padded_word) - k + 1):
            k_gram = padded_word[i:i+k]
            k_grams.add(k_gram)
        
        return k_grams

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of k-grams.
        second_set : set
            Second set of k-grams.

        Returns
        -------
        float
            Jaccard score.
        """
        if not first_set and not second_set:
            return 1.0
        if not first_set or not second_set:
            return 0.0
        
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        
        return intersection / union

    def k_gramming_and_counting(self, all_documents):
        """
        k-grams all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_k_gram_words : dict
            A dictionary from words to their k-grams sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_k_gram_words = {}
        word_counter = Counter()
        
        for doc in all_documents:
            words = doc.split()
            for word in words:
                word_counter[word] += 1
                
                if word not in all_k_gram_words:
                    all_k_gram_words[word] = self.k_gram_word(word, 2)
        
        return all_k_gram_words, dict(word_counter)

    def save(self, path):
        """
        Save the k-grams data and word counter to a file.
        """
        data = {
            'all_k_gram_words': self.all_k_gram_words,
            'word_counter': self.word_counter
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        """
        Load the shingle data and word counter from a file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.all_k_gram_words = data['all_k_gram_words']
            self.word_counter = data['word_counter']

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        word_k_grams = self.k_gram_word(word, 2)
        
        candidates = {}
        
        for vocab_word, vocab_k_grams in self.all_k_gram_words.items():
            jaccard = self.jaccard_score(word_k_grams, vocab_k_grams)
            if jaccard > 0:
                candidates[vocab_word] = jaccard
        
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:10]
        
        scored_candidates = []
        for cand_word, jaccard_score in sorted_candidates:
            normalized_tf_score = min(1.0, self.word_counter.get(cand_word, 0) / max(self.word_counter.values()))
            final_score = jaccard_score * 0.6 + normalized_tf_score * 0.4
            scored_candidates.append((cand_word, final_score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [cand[0] for cand in scored_candidates[:5]]

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        words = query.split()
        corrected_words = []
        
        for word in words:
            if word in self.word_counter:
                corrected_words.append(word)
            else:
                nearest = self.find_nearest_words(word)
                if nearest:
                    corrected_words.append(nearest[0])
                else:
                    corrected_words.append(word)
        
        return ' '.join(corrected_words)