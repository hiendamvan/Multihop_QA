import os
import json
import re 
from tqdm import tqdm
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# --- Initialization & Setup ---

def initialize_app():
    """Loads environment variables and initializes the language model."""
    load_dotenv()
    llm = ChatOpenAI(model= 'gpt-4.1-nano', 
                    openai_api_key=os.getenv('') )
    return llm

# --- Data Processing ---

def load_corpus(file_path: str) -> list[Document]:
    """
    Reads a corpus from a file, splits it into documents, 
    and converts them into Document objects.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    
    # Split the corpus by the "Title:" delimiter
    raw_docs = corpus.split("Title:")
    docs = []
    for raw_doc in raw_docs:
        raw_doc = raw_doc.strip()
        if raw_doc:
            lines = raw_doc.split("\n", 1)
            title = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ""
            # Create a Document object with content and metadata
            docs.append(Document(page_content=f"{title}\n{content}", metadata={"title": title}))
    return docs

def load_qa_dataset(file_path: str) -> tuple[list, list, list]:
    """Loads questions, answers, and question types from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract data from the JSON structure
    questions = [item['query'] for item in data]
    answers = [item['answer'] for item in data]
    question_types = [item['question_type'] for item in data]
    
    return questions, answers, question_types

# --- Chain and Retriever Creation ---

def create_chains(llm):
    """Creates the processing chains for sub-question generation, reasoning, and final answer."""
    # Chain to generate an initial sub-question
    subq_prompt = PromptTemplate.from_template(
        "Given the question: '{question}', what would be a good sub-question to answer first?"
    )
    subq_chain = subq_prompt | llm

    # Chain to reason about the next step based on retrieved context
    reasoning_prompt = PromptTemplate.from_template(
        "Original question: '{orig_question}'\n"
        "Current sub-question: '{sub_question}'\n"
        "retrieved context:\n{context}\n\n"
        "What should the next sub-question be, or should we attempt to answer the original question now?"
    )
    reasoning_chain = reasoning_prompt | llm

    # Chain to generate the final answer based on the conversation history
    final_prompt = PromptTemplate.from_template(
        "We are answering the question: '{question}'.\n"
        "We have gone through the following reasoning and retrieved steps:\n"
        "{history}\n"
        "Based on the above reasoning and retrieved context, provide the final answer concisely and directly.\n"
        "- If the question is factual (e.g Who is the author of Gone with the wind?), response with only the answer(e.g Margaret Mitchell)\n"
        "- If the question is Yes/ No question, just answer Yes or no\n"
        "- If the information is insufficient to answer, response with: 'Insufficient information'."
        "**Do not explain your reasoning or repeat the question.**"
    )
    final_chain = final_prompt | llm
    
    return subq_chain, reasoning_chain, final_chain

def create_hybrid_retriever(docs: list[Document], persist_directory: str = "./chroma_db") -> EnsembleRetriever:
    """Creates a hybrid retriever combining BM25 and an embedding-based search (Chroma)."""
    # 1. Setup BM25 (Sparse Retriever) for keyword-based search
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3 # Retrieve top 4 results

    # 2. Setup Chroma (Dense Retriever) for semantic search
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if a vector store already exists to avoid re-creating it
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        print("✅ Loaded vector store from disk.")
    else:
        print("⚙️ Generating and storing embeddings into vector store...")
        os.makedirs(persist_directory, exist_ok=True)
        # Create and persist the vector store
        vector_store = Chroma.from_documents(
            documents=docs, 
            embedding=embedding_model, 
            persist_directory=persist_directory
        )
        print("✅ Done embedding and saving.")
    
    dense_retriever = vector_store.as_retriever(search_kwargs={'k': 3}) # Retrieve top 4 results

    # 3. Create the Ensemble Retriever to combine both sparse and dense results
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever], 
        weights=[0.5, 0.5] # Give equal weight to both retrievers
    )
    
    return ensemble_retriever

# Normalize answer and query for evaluation 

def normalize_answer(s: str) -> str:
    """Chuẩn hóa câu trả lời: viết thường, loại bỏ dấu câu, khoảng trắng"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# Evaluation: F1-score between predicted answer and ground truth 

def compute_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    common = set(pred_tokens) & set(gt_tokens)
    
    if(len(common) == 0):
        return 0.0 
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1_score = 2 * precision * recall / (precision + recall)
    
    return f1_score

# --- Main Execution Flow ---

def run_ircot_multihop(query: str, retriever: EnsembleRetriever, chains: tuple, max_hops: int = 3) -> str:
    """
    Executes the multi-hop reasoning process (Iterative Reasoning and Context Retrieval)
    to answer a complex question.
    """
    subq_chain, reasoning_chain, final_chain = chains
    history = []
    current_query = query

    for hop in range(max_hops):
        #print(f"\n➡️ Hop {hop+1}: Reasoning on '{current_query}'")

        # Step 1: Generate a sub-question
        subq = subq_chain.invoke({"question": current_query}).content.strip()
        print(f"🧠 Sub-question: {subq}")

        # Step 2: Retrieve relevant documents for the sub-question
        retrieved_docs = retriever.invoke(subq)
        context = "\n\n".join([d.page_content for d in retrieved_docs])
        print(f"📄 Retrieved context:\n{context}...")  # Print first 500 chars of context
        
        # Step 3: Reason about the next step or decide to answer
        next_query = reasoning_chain.invoke({
            "orig_question": query,
            "sub_question": subq,
            "context": context
        }).content.strip()

        # Store the sub-question and its context in the history
        history.append((subq, context))
        current_query = next_query
        
        # Early stopping condition if the model decides it has enough info
        if "answer the original question now" in next_query.lower():
            break

    # Build the history text for the final prompt
    hist_text = ""
    for i, (subq, ctx) in enumerate(history):
        hist_text += f"Step {i+1}:\nSub-question: {subq}\nContext:\n{ctx[:500]}...\n\n"

    # Step 4: Generate the final answer
    final_answer = final_chain.invoke({
        "question": query,
        "history": hist_text
    }).content.strip()

    #print("\n✅ Final Answer:", final_answer)
    return final_answer


# --- Main Execution Block ---

# This block will only run when the script is executed directly
if __name__ == "__main__":
    # 1. Initialize the application (LLM)
    llm = initialize_app()

    # 2. Load and prepare data
    documents = load_corpus("dataset/multihoprag_corpus.txt")
    questions, answers, question_types = load_qa_dataset('dataset/MultiHopRAG.json')

    # 3. Create the retriever and processing chains
    hybrid_retriever = create_hybrid_retriever(documents)
    all_chains = create_chains(llm)

    f1_scores = []
    # 4. Test with 100 questions
    for i in range(54,55):
        query = questions[i]
        ground_truth = normalize_answer(answers[i])
        
        #print(f"❓ Query: {query}")
        predicted_answer = run_ircot_multihop(
            query=query, 
            retriever=hybrid_retriever, 
            chains=all_chains
        )
        predicted_answer = normalize_answer(predicted_answer)
        #print("🎯 Ground truth answer:", ground_truth)
        f1_scores.append(compute_f1_score(predicted_answer, ground_truth))
        if(normalize_answer(predicted_answer) !=  normalize_answer(ground_truth)):
            print(f'Question {i} false, question type is {question_types[i]}.')
            print('Predicted answer:', predicted_answer)
            print('Ground truth:', ground_truth)
        print(f'✅ Question {i} done.')
    avg = sum(f1_scores) / len(f1_scores)
    print(f"Average F1-score: {avg:.2f}")