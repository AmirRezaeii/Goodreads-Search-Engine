import string
from collections import deque
from typing import Callable, List, Tuple, Dict

class Snippet:
    """
    A class to generate relevant text snippets from documents based on a search query.
    
    It uses a single-pass sliding window with a 'lag' mechanism to ensure keywords
    at the very beginning and very end of documents are treated as potential centers.
    """

    def __init__(self, normalize_function: Callable, remove_stopword_function: Callable, number_of_words_on_each_side: int = 5):
        """
        Initialize the Snippet generator.

        Args:
            normalize_function (Callable): A function that takes a word and returns its stemmed/normalized version.
            remove_stopword_function (Callable): A function that takes a query string and returns a list of filtered tokens.
            number_of_words_on_each_side (int): Number of words to include to the left and right of a keyword.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        self.normalize = normalize_function
        self.remove_stopword = remove_stopword_function
        self.win_size = (2 * number_of_words_on_each_side) + 1

    def find_snippet(self, raw_doc: str, query: str) -> Tuple[str, List[str]]:
        """
        Main orchestrator for snippet generation.

        Parameters:
            raw_doc (str): The original document string.
            query (str): The user's search query string.

        Returns:
            final_snippet (str): The formatted snippet with '***' highlighting and '...' separators.
            not_exist_words (list): The list of words from the query that were not found in the document.
        """
        doc_tokens = raw_doc.split()
        
        import string
        cleaned_tokens = [word.strip(string.punctuation) for word in doc_tokens]
        normalized_cache = [self.normalize(word) for word in cleaned_tokens]
        
        query_tokens = self.remove_stopword(query)
        normalized_query = [self.normalize(token) for token in query_tokens]
        query_set = set(normalized_query)
        
        found_words = set()
        for norm_token in normalized_cache:
            if norm_token in query_set:
                found_words.add(norm_token)
        
        not_exist_words = [word for word in normalized_query if word not in found_words]
        
        best_windows = self._identify_best_windows(doc_tokens, normalized_cache, query_set)
        
        merged_windows = self._merge_windows(best_windows)
        
        final_snippet = self._create_snippet_text(doc_tokens, normalized_cache, merged_windows, query_set)
        
        return final_snippet, not_exist_words


    def _identify_best_windows(self, doc_tokens: list, normalized_cache: list, query_set: set) -> List[Tuple[int, int]]:
        """
        Uses a sliding window to score the 'density' of query matches.
        
        Parameters:
            doc_tokens (list): List of original words from the document.
            normalized_cache (list): List of the same words, but normalized/stemmed.
            query_set (set): Set of normalized query stems.

        Returns:
            list: A list of (start_index, end_index) for the best windows found.
        """
        if len(doc_tokens) < self.win_size:
            return [(0, len(doc_tokens) - 1)]
        
        windows = []
        window_centers = []
        
        for center in range(len(doc_tokens)):
            start = max(0, center - self.number_of_words_on_each_side)
            end = min(len(doc_tokens) - 1, center + self.number_of_words_on_each_side)
            
            score = 0

            for i in range(start, end + 1):
                if normalized_cache[i] in query_set:
                    score += 1

            window_centers.append((score, center, start, end))
        
        sorted_windows = sorted(window_centers, key=lambda x: x[0], reverse=True)
    
        used_indices = set()
        for score, center, start, end in sorted_windows:
            if score == 0:
                break
            
            is_unique = True
            for i in range(start, end + 1):
                if i in used_indices:
                    is_unique = False
                    break
            
            if is_unique:
                windows.append((start, end))
                for i in range(start, end + 1):
                    used_indices.add(i)

        return windows

    def _merge_windows(self, windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Combines window ranges that overlap or touch.
        
        Parameters:
            windows (list): List of (start, end) index tuples.

        Returns:
            list: List of merged (start, end) index tuples.
        """
        if not windows:
            return []
        
        sorted_windows = sorted(windows, key=lambda x: x[0])
        
        merged = []
        current_start, current_end = sorted_windows[0]
        
        for start, end in sorted_windows[1:]:
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        
        return merged

    def _create_snippet_text(self, doc_tokens: list, normalized_cache: list, 
                             merged_windows: List[Tuple[int, int]], query_set: set) -> str:
        """
        Constructs the final formatted snippet string.
        
        Parameters:
            doc_tokens (list): Original document tokens.
            normalized_cache (list): Stemmed document tokens.
            merged_windows (list): Merged (start, end) indices.
            query_set (set): Normalized query stems.

        Returns:
            str: The final snippet with highlights and ellipses. 
                example: "The ***wizard*** went to ***Hogwarts.*** The ***wizard*** loved magic."

        """
        if not merged_windows:
            if len(doc_tokens) <= 20:
                return ' '.join(doc_tokens[:20])
            return ' '.join(doc_tokens[:20]) + '...'
        
        snippets = []
        
        for idx, (start, end) in enumerate(merged_windows):
            window_tokens = []
            
            for i in range(start, end + 1):
                word = doc_tokens[i]
                if normalized_cache[i] in query_set:
                    window_tokens.append(f"***{word}***")
                else:
                    window_tokens.append(word)
            
            snippet = ' '.join(window_tokens)
            snippets.append(snippet)
        
        result = ' ... '.join(snippets)
        
        if merged_windows[0][0] > 0:
            result = '... ' + result
        
        if merged_windows[-1][1] < len(doc_tokens) - 1:
            result = result + ' ...'
        
        return result