from modules.embedding.store import store_documents
from modules.splitting.splitter import recursive_split
from modules.retrieving.retrieving import get_qa_chain
docs = recursive_split()

qa_chain = get_qa_chain("recursive_chunking")

query = "Parle moi du contenu du sommaire"

print(qa_chain.invoke(query))

#store_documents(docs=docs,collection_name="base_chunking")
