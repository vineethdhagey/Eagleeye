import argparse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, required=True, help="Path to FAISS index directory")
    parser.add_argument("--query", type=str, required=True, help="Your natural language question")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model name")
    parser.add_argument("--top_k", type=int, default=5, help="How many chunks to retrieve")
    args = parser.parse_args()

    # Load embeddings
    print("🔹 Loading FAISS index...")
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(args.index, emb, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": args.top_k})

    # Load small LLM (TinyLlama or another HF chat model)
    print(f"🔹 Loading model {args.model} (this may take a while)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Prompt template for RAG
    template = """
    You are a maritime logistics assistant. Use ONLY the retrieved context to answer.
    If you don't find an answer, say "insufficient data."

    Question: {question}

    Context:
    {context}

    Answer in plain English with numbers if possible.
    """
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    print("🔹 Running query...")
    result = qa.invoke(args.query)

    print("\n✅ Answer:", result["result"])
    print("✅ Sources: retrieved chunks from FAISS")

if __name__ == "__main__":
    main()
