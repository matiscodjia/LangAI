import chromadb
from chromaviz import visualize_collection

client = chromadb.PersistentClient(path="../data/chromaDB")
collection = client.get_collection("specific")
visualize_collection(collection)
