from typing import List, Dict

class Retriever:
    def __init__(self, embedder, vector_store, chunks,max_distance=2.5):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks = chunks
        self.max_distance = max_distance

    def retrieve(self,query: str,k: int = 5) -> List[Dict]:

        query_embedding = (self.embedder.embed_query(query))
        

        distances, indices = (self.vector_store.search(query_embedding,k))
        
        print("\nDistances:", distances[0])

        results = []

        for distance,idx in zip(distances[0],indices[0]):
            if distance>self.max_distance:
                continue
            
            results.append(self.chunks[idx])

        return results