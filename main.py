import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import openai

load_dotenv()

openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama3-8b-8192"


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "book-recommender1")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


pc = Pinecone(api_key=PINECONE_API_KEY)


if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX)


df = pd.read_csv("book_dataset_20.csv")
df["combined"] = (
    df["title"] + " " + df["author"] + " " + df["genre"] + " " +
    df["mood"] + " " + df["description"] + " " + df["keywords"]
)

embedder = SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text):
    return embedder.encode(text).tolist()


def upload_books_to_pinecone():
    print("Uploading book embeddings to Pinecone...")
    vectors = []
    for i, row in df.iterrows():
        vector_id = f"book-{i}"
        metadata = {
            "title": row["title"],
            "author": row["author"],
            "description": row["description"],
            "year": str(row["year"])
        }
        embedding = get_embedding(row["combined"])
        vectors.append((vector_id, embedding, metadata))
    index.upsert(vectors)
    print("Upload complete.")


if index.describe_index_stats().get("total_vector_count", 0) == 0:
    upload_books_to_pinecone()


def recommend_books(user_text, top_n=3):
    user_emb = get_embedding(user_text)
    query_response = index.query(
        vector=user_emb,
        top_k=top_n,
        include_metadata=True
    )

    valid_titles = set(df["title"].tolist())
    recs = []

    for match in query_response["matches"]:
        md = match.get("metadata", {})
        title = md.get("title", "")
        if title in valid_titles:
            recs.append({
                "title": title,
                "author": md.get("author", ""),
                "description": md.get("description", ""),
                "year": md.get("year", "")
            })

    return pd.DataFrame(recs)



def get_ai_reply(history):
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=history
    )
    return response['choices'][0]['message']['content']


def run_chatbot():
    print("📚 AI Book Recommender Bot")
    print("Tell me about your preferences. When you're ready, type 'recommend'. Type 'exit' to quit.\n")

    chat_memory = [
        {"role": "system",
         "content": (
             "You are a friendly assistant that only helps users choose books from a provided dataset."
             "\n- ONLY talk about books. Do not answer questions unrelated to books."
             "\n- ONLY recommend books that are found in the book_dataset (the user vector search is your only source)."
             "\n- Do NOT recommend any books unless the user explicitly types the word 'recommend' on its own."
             "\n- call the function recommend_books() when the user types 'recommend'"
             "\n- NEVER list books without user typing 'recommend'."
             "\n- You are Tarka's Bookshop assistant"
             "\n- If the user says something like 'can you recommend me a book?', just answer with 'Yes'."
             "\n- Ask natural follow-up questions about their reading preferences (e.g. genre, mood, author)."
         )}
    ]

    full_user_text = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye! 📖")
            break
        elif user_input.strip().lower() == "recommend":
            print("\n🤖 Bot: Based on everything you told me, here are some books for you:\n")
            books = recommend_books(full_user_text)
            for _, row in books.iterrows():
                print(f"📘 {row['title']} by {row['author']} ({row['year']})")
                print(f"    {row['description']}\n")
            continue

        full_user_text += " " + user_input
        chat_memory.append({"role": "user", "content": user_input})
        reply = get_ai_reply(chat_memory)
        chat_memory.append({"role": "assistant", "content": reply})
        print(f"🤖 Bot: {reply}\n")


if __name__ == "__main__":
    run_chatbot()
