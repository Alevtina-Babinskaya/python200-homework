from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("assignments_06/ground_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
docs = SimpleDirectoryReader(docs_dir).load_data()
print("Total number of documents loaded: ", len(docs))
for d in docs:
    print(d.metadata['file_name'])

index = VectorStoreIndex.from_documents(docs)
print("Index built successfully. Ready to answer questions.")
query_engine = index.as_query_engine(similarity_top_k = 3)

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
for q in questions:
    print("\n" + "-" * 60 + "\n")
    print("Question: ", q)
    response = query_engine.query(q)
    print("Response: ", response)
    source = response.source_nodes[0]
    print(f"Document: {source.node.metadata.get('file_name', 'Unknown')}")
    print(f"Similarity Score: {source.score:.4f}")
    print(f"Text Snippet: {source.node.get_content()[:200]}...")

# The answer to the first and second questions were correct, but the retrieval results were not. 
# The system retrieved our_story.txt with a similarity score of 0.8078, even though the weekend hours are 
# listed in the FAQ/Hours section. Same for the second question.
# This shows that a high similarity score does not always mean that the most relevant chunk was retrieved. 


test_question = "What days will Groundwork Coffee be closed this year?"
test_resp = query_engine.query(test_question)
print("\n" + "-" * 60 + "\n")
print("Test question: ", test_question)
print("Assistants response: ", test_resp)
for node in test_resp.source_nodes:
    print("Document: ", node.node.metadata.get('file_name', 'Unknown'))
    print(f"Similarity score: {node.score:.4f}")
    print("Text: ", node.node.get_content()[:200])

# I expected this question to be difficult because the documents only mention Thanksgiving Day and Christmas Day, 
# but they do not provide the exact dates for these holidays in the current year. 
# The model would need additional calendar information to answer with specific dates. 
# It also needs to determine whether only the two holidays mentioned in the documents should be included or 
# whether other days, such as the Friday after Thanksgiving, should also be considered.

# The model could not answer the question using only the retrieved information. 
# The exact dates were not included in the documents. However, the model included information, that was not in the 
# documents - New Year Holiday.

# The model remained confident even though the retrieved information was insufficient to provide the exact dates. 
# In this particular case, the response failure was not very serious because the model did not invent unsupported 
# information about Groundwork's holiday schedule. However, this shows that a confident tone does not necessarily 
# mean that the answer is fully supported by the available sources. This suggests that we should be cautious about 
# trusting AI-generated responses simply because they sound confident.
# 
# I would provide the model with more reliable and relevant sources, such as an official calendar or 
# other trusted sources containing holiday dates. 
# I would also make the system clearly distinguish between information retrieved from the provided documents and 
# information obtained from outside sources.
# --------------------------------------------------------------------------------------------------------------

# The LlamaIndex version took only a few lines of code — about 2 lines if we count just creating the vectors and 
# the query engine. It really shows how much easier a framework can make things. Without it, we had to write 
# a lot more code to do the same basic steps.

# I think this could be useful for almost any business that has a lot of information about its services or products. 
# For example, banks or insurance companies could have an AI assistant that answers customer questions based on 
# their policies and other documents. It could save employees a lot of time answering the same questions over and over.

# Even when RAG finds the right information, the model can still give a wrong answer. It can misunderstand the retrieved 
# information or make something up. So, having good retrieval doesn't guarantee that the final answer will be correct.
