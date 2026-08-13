from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PyMuPDFReader
from pathlib import Path




if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# RAG Concepts
# Concepts Question 1
# Scenario A: RAG
# I would use RAG because there are hundreds of PDFs, and the information
# changes every quarter. RAG lets the model search the documents and use
# the most relevant information without having to retrain the model each time.
#
# Scenario B: Fine-tuning
# I would use fine-tuning because the main goal is to teach the model a very
# specific writing style. They already have 3,000 examples from their writers,
# which gives the model plenty of examples to learn this unique brand voice.
#
# Scenario C: Prompt engineering
# I would use prompt engineering because the report is only two pages and
# the analyst only needs to work with this one document. There is no need
# to build a RAG system or fine-tune a model when the document can simply
# be included in the prompt.

# Concepts Question 2
# A confidently wrong answer is more harmful because it makes the user
# believe that the information is correct and they may not think to check it.
#
# For example, if someone asks AI about business license requirements and
# AI confidently says that no license is needed, the person might follow
# that advice and unknowingly break the law, which could lead to fines or
# other legal consequences.
#
# The way AI expresses an answer also affects how much we trust it. If the
# answer sounds confident and certain, people are more likely to believe it,
# even when the information is actually wrong. An answer that says "I am not
# sure" signals that the information should be checked before relying on it.

# Concepts Question 3
# steps = [
#     "Extract text from source documents",        # Extracts text from the source documents.
#     "Split text into chunks",                    # Splits the text into smaller pieces.
#     "Convert text chunks into embeddings",       # Converts each chunk into a vector of numbers
#                                                  # that represents its meaning.
#     "Receive the user's query",                  # Receives the user's question.
#     "Embed the user's query",                    # Converts the user's question into an embedding
#                                                  # using the same embedding model.
#     "Retrieve the most relevant chunks",         # Compares the query embedding with the chunk
#                                                  # embeddings and finds the most relevant chunks.
#     "Inject retrieved chunks into the prompt",   # Adds the relevant chunks to the prompt along
#                                                  # with the user's question and sends it to the LLM.
#     "Generate a response from the LLM",          # The LLM uses the question and retrieved context
#                                                  # to generate an answer.
# ]

# Keyword RAG
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
# Keyword Question 1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
print("\nKeyword Question 1:", query)
result = simple_keyword_retrieval(query, documents, verbose=True)
print("\nAnswer:")
print(result[0][1])
# loyalty.txt was selected as a best match, because 3 files of 4 has the same overlap score - 1. When "next" compares matches it first looks at scores. 
# If scores are equal, it compares names. With reverse=True, loyalty.txt alphabetically comes first. 

# Keyword Question 2
query = "Do you have anything without caffeine?"
print("\nKeyword Question 2:", query)
simple_keyword_retrieval(query, documents, verbose=True)
print("\nAnswer:")
print(result[0][1])
# No documents were selected. menu.txt includes the information about coffee choices, 
# but text doesn't contain the exact word "caffeine" - 0 overlaps.
# Semantic RAG would do better here, because it takes the words meaning into account, not just counts overlaps in words.

# Keyword Question 3
query = "How do I sign up for rewards?"
print("\nKeyword Question 3:", query)
# Model won't select any documents because the only overlap in text is "for" which is excluded as stopword from text.
simple_keyword_retrieval(query, documents, verbose=True)
print("\nAnswer:")
print(result[0][1])
# I predicted zero overlap and my prediction was correct. 

# Semantic RAG Concepts
# Semantic Question 1

# A vector embedding is a representation of text as a set of numbers that
# captures its meaning. Texts with similar meanings tend to have similar vectors.

# The chunk with a cosine similarity score of 0.85 is more relevant. The
# higher score means that the meaning of the chunk is more similar to the query.

# Semantic search can find a relevant chunk even when the exact words do not
# appear because it compares the meanings of the texts using their vector
# embeddings, rather than just looking for matching words.

# Semantic Question 2
# | Feature                    | Keyword RAG                       | Semantic RAG                    |
# |----------------------------|-----------------------------------|---------------------------------|
# | What is compared?          | Exact word overlap                | vector embeddings               |
# | What is retrieved?         | Full document                     | relevant chunks of text         |
# | Can it handle synonyms?    | No                                | Yes                             |
# | Storage format             | Plain text dictionary             | text chunks + embedding vectors |
# | Relevance score            | Number of overlapping keywords    | Similarity score                |

# LlamaIndex
# LlamaIndex Question 1
# PG_HOST = 'localhost'
# PG_PORT = "5432"
# PG_DATABASE = "ctd_rag"
# PG_USER = "ctd"
# PG_PASSWORD = "ctdpassword"
# LI_TABLE_NAME = "li_brightleaf_pgvector"

# EMBED_MODEL_NAME = "text-embedding-3-small"
# EMBED_DIM = 1536
# PDF_DIR = "brightleaf_pdfs"
# BUILD_INDEX = True

# Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL_NAME)
# vector_store_q = PGVectorStore.from_params(
#     host=PG_HOST,
#     port=PG_PORT,
#     database=PG_DATABASE,
#     user=PG_USER,
#     password=PG_PASSWORD,
#     table_name=LI_TABLE_NAME,
#     embed_dim=EMBED_DIM,
# )
# index_q = VectorStoreIndex.from_vector_store(vector_store=vector_store_q)
# query_engine = index_q.as_query_engine(similarity_top_k = 3)
docs = []
pdf_folder = Path("assignments_06/brightleaf_pdfs")
reader = PyMuPDFReader()
for pdf_file in pdf_folder.glob("*.pdf"):
    doc = reader.load(str(pdf_file))
    docs.extend(doc)

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k = 3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)
# Q1:
# The retrieved chunks look relevant to the question because they contain
# information about employee benefits. The model's response sounds confident
# and specific and does not use phrases like "I'm not sure."
# I did not notice anything unexpected in the retrieved chunks.

# Q2:
# The retrieved chunks look relevant because they contain information about
# BrightLeaf's security policies. The model also sounds confident and specific.
# Some less relevant chunks were retrieved even though they had relatively
# high similarity scores.

# LlamaIndex Question 2
query_engine_1 = index.as_query_engine(similarity_top_k=1)
query_engine_5 = index.as_query_engine(similarity_top_k=5)
question =  "What employee benefits does BrightLeaf offer?"
response_1 = query_engine_1.query(question)
print("k1 response:", response_1)
for node_with_score in response_1.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")

response_5 = query_engine_5.query(question)
print("k5 response:", response_5)
for node_with_score in response_5.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
      
# With top_k=1, the model gets only the most relevant chunk, so the answer
# may be more focused but could miss some information. With top_k=5, the
# model gets more context, which may make the answer more complete, but less relevant.

# LlamaIndex Question 3
test_question = "Who owns shares of BrightLeaf?"
response_test = query_engine.query(test_question)
print("Question:", test_question)
print("\nResponse:", response_test)

print("\nRetrieved chunks:")
for node_with_score in response_test.source_nodes:
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text corrected: {node_with_score.node.get_content()[:150]}")
    print("-" * 50)
# I expected the pipeline to struggle because the documents may not contain
# information about who owns BrightLeaf's shares. The system 
# retrieved chunks that are somewhat related but do not contain the accurate information.
# The response doesn't include any specific companies and names, just a vague information that seems unreliable.
# To handle this better, I would instruct the model to say when the information is not available instead of guessing.

# LlamaIndex Question 4
q = "What employee benefits does BrightLeaf offer?"
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
faithfulness_ev = FaithfulnessEvaluator(llm=llm)
relevancy_ev = RelevancyEvaluator(llm=llm)
response_ev = query_engine.query(q)
faithfulness_result = faithfulness_ev.evaluate_response(query=q, response=response_ev)
print("Faithfulness Evaluation Q1: " + str(faithfulness_result.score))
relevancy_result = relevancy_ev.evaluate_response(query=q, response=response_ev)
print("Relevancy Result Q1: " + str(relevancy_result.score))
q1 = "What is the best coffeeshop near BrightLeaf office?"
response_ev1 = query_engine.query(q1)
faithfulness_result = faithfulness_ev.evaluate_response(query=q1, response=response_ev1)
print("Faithfulness Evaluation Q2: " + str(faithfulness_result.score))
relevancy_result = relevancy_ev.evaluate_response(query=q1, response=response_ev1)
print("Relevancy Result Q2: " + str(relevancy_result.score))
# A faithfulness score of 1.0 means that the response is fully supported
# by the retrieved context. A score of 0.0 means that the response is not
# supported by the retrieved context.

# Relevancy measures how well the response answers the user's question.
# Faithfulness is about whether the answer is supported by the retrieved
# information, while relevancy is about whether the answer actually
# addresses the question. 

# The scores for the two questions are different because the response to the first question 
# can be easily retrieved from the context. The response for the second question is most likely missing
# from the provided context. 

# The "LLM-as-a-judge" approach is a way to evaluate the response using an external LLM. 
# It is used for semantic RAG evaluation instead of accuracy metrics because RAG uses semantic 
# meaning instead of numbers and categories 
