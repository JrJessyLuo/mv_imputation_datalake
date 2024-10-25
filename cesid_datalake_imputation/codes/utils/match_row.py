import heapq
import numpy as np
import pandas as pd
from nltk import ngrams
import time
from sentence_transformers import SentenceTransformer
# import faiss
import glob
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


def tuple_serializer(tuple_t, target_att=None): 
    """Returns string of serialized tuple."""
    if target_att is not None:
        all_cols = [k for k in tuple_t.keys() if k != target_att]
    else:
        all_cols = tuple_t.keys()
    arr = [(col.replace('\r\n', ' ').replace('\n',  ' '), str(tuple_t[col]).replace('\r\n', ' ').replace('\n',  ' ')) for col in all_cols]
    serialization = (' ; ').join([f'{pair[0]} : {pair[1]}' for pair in arr])
    if target_att is not None:
        serialization = f'{serialization} ; {target_att} : '
    serialized_tuple = f'[ {serialization} ]'
    return serialized_tuple

def preprocess_and_embed_rows(table, query_cols, target_att=None):
    """Preprocess and compute embeddings for the rows in a table."""
    # Drop duplicates and keep track of the original indices
    simplified_table = table.iloc[:, query_cols].drop_duplicates()
    serialized_rows = [tuple_serializer(row, target_att) for _, row in simplified_table.iterrows()]
    embeddings = model.encode(serialized_rows)
    return np.array(embeddings), simplified_table.index.tolist()

def is_missing_value(val):
    """Check if a value is missing (None or np.nan)."""
    return val is None or (isinstance(val, float) and np.isnan(val))

def is_numeric_string(val):
    """Check if a string represents a numerical value."""
    try:
        float(val)
        return True
    except ValueError:
        return False

def jaccard_similarity(str1, str2, q=2):
    """Calculate Jaccard similarity between two strings based on q-grams."""
    set1 = set(ngrams(str1, q))
    set2 = set(ngrams(str2, q))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Unified function for calculating similarity and bounds
def calculate_similarity_and_bounds(query_tuple, candidate_tuple):
    similarity = 0
    lower_bound = 0
    upper_bound = 0
    num_comparable = 0

    for val1, val2 in zip(query_tuple, candidate_tuple):
        if not is_missing_value(val1) and not is_missing_value(val2):  # Non-missing values
            if is_numeric_string(val1) and is_numeric_string(val2):  # Numerical comparison
                score = 1 - abs(float(val1) - float(val2)) / max(float(val1), float(val2), 1)
            else:  # String comparison (Jaccard)
                score = jaccard_similarity(str(val1), str(val2))

            similarity += score
            lower_bound += score
            upper_bound += score
            num_comparable += 1
        elif is_missing_value(val1) or is_missing_value(val2):  # If one or both values are missing
            upper_bound += 1  # Assume max similarity for the missing value

    # Normalize the results based on the number of comparisons
    if num_comparable > 0:
        similarity /= num_comparable
        lower_bound /= num_comparable
    upper_bound /= (num_comparable + 1)  # Account for missing values in upper bound

    return similarity, lower_bound, upper_bound

# Insert into top-k heap with a fixed size
def insert_into_topk(topk_heap, candidate_tuple, score, row_index, k):
    # Convert candidate_tuple into a tuple (so it's hashable and works well with heap)
    candidate_tuple = tuple(candidate_tuple)

    # print('--', score, type(score), candidate_tuple, row_index)
    
    if len(topk_heap) < k:
        heapq.heappush(topk_heap, (score, candidate_tuple, row_index))
    else:
        heapq.heappushpop(topk_heap, (score, candidate_tuple, row_index))


"""
    calculcate the row-level similarities
"""
def find_topk_similar_tuples(query_tuple, candidate_tuples, k):
    topk_heap = []  # Min-heap for top-k tuples, keeps the worst element on top

    for idx, candidate_tuple in candidate_tuples.iterrows():
        # Convert candidate_tuple from Series to list for comparison
        candidate_tuple_list = candidate_tuple.values
        
        # Calculate the exact similarity and bounds in one step
        exact_similarity, lower_bound, upper_bound = calculate_similarity_and_bounds(query_tuple, candidate_tuple_list)
        # print('--', exact_similarity, type(exact_similarity), lower_bound, type(lower_bound), upper_bound, type(upper_bound))

        
        # If the upper bound is smaller than the worst score in top-k, discard
        if len(topk_heap) == k and upper_bound <= topk_heap[0][0]:
            continue
        
        # If the lower bound is larger than the worst score, insert without exact similarity
        if len(topk_heap) < k or lower_bound > topk_heap[0][0]:
            # print('--', topk_heap, lower_bound)
            try:
                insert_into_topk(topk_heap, candidate_tuple_list, lower_bound, idx, k)
            except:
                print('error')
            continue

        # Insert the exact similarity if it's better than the worst score in the heap
        if len(topk_heap) < k or exact_similarity > topk_heap[0][0]:
            insert_into_topk(topk_heap, candidate_tuple_list, exact_similarity, idx, k)
    return topk_heap
    # print([(type(_[0]),_[0]) for _ in topk_heap])
    # return sorted(topk_heap, reverse=True)  # Return sorted top-k tuples

# def find_topk_similar_tuples_byembeddings(query_tuple, table, query_cols, k):
#     candidate_embeddings = preprocess_and_embed_rows(table, query_cols, target_att=None)[0]

#     query_serialized = tuple_serializer(query_tuple)
#     query_embedding = model.encode([query_serialized])[0]

#     faiss_index = faiss.IndexFlatL2(query_embedding.shape[0])
#     faiss_index.add(candidate_embeddings)
            
#     _, indices = faiss_index.search(np.array([query_embedding]), k)
#     return indices