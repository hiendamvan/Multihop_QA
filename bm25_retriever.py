from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
import json

from dotenv import load_dotenv
import os
load_dotenv()

# read corpus dataset 
corpus = ""
with open("dataset/multihoprag_corpus.txt", "r", encoding="utf-8") as f:
    corpus = f.read()
    
# split corpus by title 
raw_docs = corpus.split("Title:")

# convert raw docs to Document 
docs = []
for i, raw_doc in enumerate(raw_docs):
    raw_doc = raw_doc.strip()
    if raw_doc:  
        lines = raw_doc.split("\n", 1)
        title = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        full_text = f"{title}\n{content}"
        docs.append(Document(page_content=full_text, metadata={"title": title}))

# llm 
llm = ChatCohere(cohere_api_key=os.getenv('COHERE_API_KEY'), model='command-a-03-2025')

# create sub question chain 
subq_prompt = PromptTemplate.from_template(
    "Given the question: '{question}', what would be a good sub-question to answer first?"
)
subq_chain = subq_prompt | llm

# Reasoning chain
reasoning_prompt = PromptTemplate.from_template(
    "We are answering: '{orig_question}'\n"
    "Given the current sub-question: '{sub_question}'\n"
    "And the retrieved context:\n{context}\n\n"
    "What information should we retrieve information from corpus?"
)

reasoning_chain = reasoning_prompt | llm

 # final answer chain 
final_prompt = PromptTemplate.from_template(
    "We are answering the question: '{question}'.\n"
    "We have gone through the following steps:\n"
    "{history}\n"
    "Based on the above reasoning and retrieved context, answer directly, dont need to explain."
    "For example, if the question is Who is the author of Gone with the wind?. The answer is only: Margaret Mitchell"
    "If the question is Yes/ No question, just answer Yes or no"
    "If there isn't sufficient information, just answer Insufficient information."
)
final_chain = final_prompt | llm

# create retriever, return 2 most relevant document
retriever = BM25Retriever.from_documents(docs)
retriever.k = 5


def ircot_multihop(query, max_hops=2):
    history = []
    current_query = query

    for hop in range(max_hops):
        print(f"\n➡️ Hop {hop+1}: Reasoning on '{current_query}'")

        # Step 1: Generate sub-question
        subq = subq_chain.invoke({"question": current_query}).content.strip()
        #print(f"🧠 Sub-question: {subq}")

        # Step 2: retrieve  sub-question
        docs = retriever.get_relevant_documents(subq)
        context = "\n\n".join([d.page_content for d in docs])

        #print(f"📚 Retrieved context ({len(docs)} docs):")
        for i, d in enumerate(docs):
            pass
            #print(f"--- Doc {i+1} ---\n{d.metadata['title']}\n{d.page_content[:200]}...\n")

        # Step 3: Continuing reasoning
        next_query = reasoning_chain.invoke({
            "orig_question": query,
            "sub_question": subq,
            "context": context
        }).content.strip()

        history.append((subq, context))
        current_query = next_query

     # Build history text
    hist_text = ""
    for i, (subq, ctx) in enumerate(history):
        hist_text += f"Step {i+1}:\nSub-question: {subq}\nContext:\n{ctx[:500]}...\n\n"

    # final answer
    final_answer = final_chain.invoke({
        "question": query,
        "history": hist_text
    }).content.strip()
    print("\n✅ Final Answer:", final_answer)
    return final_answer

# Mở và đọc file JSON
with open('dataset/MultiHopRAG.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = []
answers = []
question_types = []

for i in range(len(data)):
    questions.append(data[i]['query'])
    question_types.append(data[i]['question_type'])
    answers.append(data[i]['answer'])

ircot_multihop(questions[2])
print("Ground truth answer", answers[2])