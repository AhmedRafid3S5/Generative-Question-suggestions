from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import json
import numpy as np
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
import numpy as np
import lancedb as LB
import pandas as pd
import os
import threading
import pyarrow as pa
from agno.storage.sqlite import SqliteStorage
from agno.memory.v2.db.sqlite import SqliteMemoryDb
import networkx as nx
import uuid

from agno.memory.v2.memory import Memory

match_threshold = 0.75
discard_threshold = 0.3



embedder = OpenAIEmbedder(api_key=OPENAI_API_KEY)

user_id = "ava"
# Database file for memory and storage
db_file = "tmp/agent.db"

question_tone = """Your questioning style follows these core principles:

**Question Structure & Style:**
- Prioritize open-ended questions that invite detailed responses and storytelling over yes/no questions
- Craft questions that are concise and avoid leading language and unnecessary complexity
- Frame questions with sufficient context but dont make very long questions
- Use phrases like "Can you walk us through..." "What does the process look like when..." "How do you approach..." instead of simple "Do you..." or "Are you..." questions

**Content Focus:**
- Center all questions around the guest's professional expertise, business ventures, creative projects, or passionate hobbies
- Demonstrate genuine interest in their specific industry, craft, or area of specialization
- Explore the nuances and complexities of their work rather than surface-level topics

**Conversational Flow:**
- Occasionally reference information shared earlier in the conversation to show active listening and create narrative threads
- When referencing past knowledge, ensure it directly connects to and enhances the current discussion topic
- Limit follow-up questions to truly compelling moments - use them strategically rather than frequently
- Allow guests space to fully develop their thoughts before introducing new directions

**Audience Engagement:**
- Align your questions with what would genuinely interest both the guest and your audience
- Consider the guest's professional background when crafting questions to ensure relevance and depth
- Ask questions that would help listeners understand both the "what" and the "why" behind the guest's work"""

# Initialize memory.v2
# memory = Memory(
#     # Use any model for creating memories
#     model=OpenAIChat(id="gpt-4.1"),
#     db=SqliteMemoryDb(table_name="user_memories", db_file=db_file),
# )
# Initialize storage
storage = SqliteStorage(table_name="agent_sessions", db_file=db_file)

agent = Agent(
        model=OpenAIChat(id="gpt-4.1"),
        update_knowledge=True,
        session_id=user_id,
        user_id=user_id,
        read_chat_history=True,

        # memory=memory,
        # enable_agentic_memory=True,
        # enable_user_memories=True,

        # Store the chat history in the database
        storage=storage,
        # Add the chat history to the messages
        add_history_to_messages=True,
        # Number of history runs
        num_history_runs=3,     
        # Adds the current date and time to the instructions
        add_datetime_to_instructions=True,
        # Adds the history of the conversation to the messages
        markdown=True,
        debug_mode=False
    )

# LanceDB setup for two tables
def get_or_create_table(db_path, table_name, schema, dummy_row):
    os.makedirs(db_path, exist_ok=True)
    db = LB.connect(db_path)
    if table_name not in db.table_names():
        df = pd.DataFrame([dummy_row])
        table = db.create_table(table_name, df, schema=schema)
        table.delete(f'id == "{dummy_row["id"]}"')
    else:
        table = db.open_table(table_name)
    return table

# Define the schema for your tables
embedding_dim = 1536  # or 384, adjust as needed
user_schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("context", pa.string()),
    pa.field("source", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
])

# Dummy row for schema creation
user_dummy = {
    "id": "dummy",
    "context": "dummy",
    "source": "dummy",
    "embedding": [0.0] * embedding_dim
}

transcribe_table = get_or_create_table("tmp/user_lancedb", "user_docs", user_schema, user_dummy)
discarded_table = get_or_create_table("tmp/skipped_lancedb", "skipped_docs", user_schema, user_dummy)

# Create FTS index ONCE at startup
transcribe_table.create_fts_index("context", replace=True, use_tantivy=True)
discarded_table.create_fts_index("context", replace=True, use_tantivy=True)

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

app = FastAPI()

class Record(BaseModel):
    id: str
    context: Optional[str] = ""

class QueryRequest(BaseModel):
    query: str

class TranscriptionRequest(BaseModel):
    transcription_text: str
    top_k: Optional[int] = 3

@app.post("/add_user_data")
def add_user_data(records: List[Record]):
    docs = []
    for rec in records:
        context = (rec.context or "").lower()
        emb = embedder.get_embedding(context)
        if not emb or len(emb) != 1536:
            continue  # skip or handle error
        docs.append({
            "id": str(rec.id),
            "context": context,
            "source": "user_info",
            "embedding": emb
        })
    for doc in docs:
        transcribe_table.delete(f'id == "{doc["id"]}"')
    transcribe_table.add(docs)
    return {"status": "success", "added": len(records)}



@app.post("/add_discarded_data")

def add_discarded_data(records: List[Record]):
    docs = []
    for rec in records:
        context = (rec.context or "").lower()
        emb = embedder.get_embedding(context)
        doc_id = str(uuid.uuid4())
        docs.append({
            "id": doc_id,
            "context": context,
            "source": "skipped_info",
            "embedding": emb
        })
    for doc in docs:
        discarded_table.delete(f'id == "{doc["id"]}"')
    discarded_table.add(docs)
    return {"status": "success", "added": len(records)}

@app.post("/delete_user_data")
def delete_user_data(ids: List[str]):
    for id_ in ids:
        try:
            transcribe_table.delete(f'id == "{id_}"')
        except Exception as e:
            return {"status": "error", "error": str(e)}
    return {"status": "success", "deleted": ids}

@app.post("/delete_discarded_data")
def delete_discarded_data(ids: List[str]):
    for id_ in ids:
        try:
            discarded_table.delete(f'id == "{id_}"')
        except Exception as e:
            return {"status": "error", "error": str(e)}
    return {"status": "success", "deleted": ids}

@app.post("/update_user_data")
def update_user_data(record: Record):
    context = (record.context or "").lower()
    emb = embedder.get_embedding(context)
    doc = {"id": record.id, "context": context, "source": "user_info", "embedding": emb}
    transcribe_table.delete(f'id == "{doc["id"]}"')
    transcribe_table.add([doc])
    return {"status": "success", "updated_id": record.id}

@app.post("/update_discarded_data")
def update_discarded_data(record: Record):
    context = (record.context or "").lower()
    emb = embedder.get_embedding(context)
    doc = {"id": record.id, "context": context, "source": "skipped_info", "embedding": emb}
    discarded_table.delete(f'id == "{doc["id"]}"')
    discarded_table.add([doc])
    return {"status": "success", "updated_id": record.id}

def rag_query(query):

    def get_top_discarded_matches(query, k=6, match_threshold=match_threshold):
        query = query.lower()
        query_emb = embedder.get_embedding(query)
        results = (
            discarded_table
            .search(query_type="hybrid", vector_column_name="embedding", fts_columns="context")
            .vector(query_emb)
            .text(query)
            .limit(10)
            .to_pandas()
        )
        results["sim"] = results["embedding"].apply(lambda emb: cosine_similarity(query_emb, emb))
        filtered = results[results["sim"] > match_threshold]
        return filtered.head(k).to_dict(orient="records")


    def get_top_user_matches(query, k=6, match_threshold=match_threshold):
        query = query.lower()
        query_emb = embedder.get_embedding(query)
        results = (
            transcribe_table
            .search(query_type="hybrid", vector_column_name="embedding", fts_columns="context")
            .vector(query_emb)
            .text(query)
            .limit(10)
            .to_pandas()
        )
        results["sim"] = results["embedding"].apply(lambda emb: cosine_similarity(query_emb, emb))
        filtered = results[results["sim"] > match_threshold]
        return filtered.head(k).to_dict(orient="records")


    top_discarded = get_top_discarded_matches(query= query.lower(), k=6)
    top_user = get_top_user_matches(query= query.lower(), k=6)


    def filter_user_chunks(user_docs, discarded_docs, discard_threshold=discard_threshold):
        filtered = []
        ids_to_delete = []
        if not user_docs or not discarded_docs:
            return user_docs
        for u_doc in user_docs:
            u_emb = u_doc.get("embedding")
            if u_emb is None:
                filtered.append(u_doc)
                continue
            u_emb = np.array(u_emb)
            if u_emb.size == 0:
                filtered.append(u_doc)
                continue
            delete_flag = False
            for d_doc in discarded_docs:
                d_emb = d_doc.get("embedding")
                if d_emb is None:
                    continue
                d_emb = np.array(d_emb)
                if d_emb.size == 0:
                    continue
                sim = cosine_similarity(u_emb, d_emb)
                print(f"Comparing {u_doc['id']} to {d_doc['id']}: similarity={sim}")
                if sim > discard_threshold:
                    print(f"Marking {u_doc['id']} for deletion (sim={sim})")
                    ids_to_delete.append(u_doc['id'])
                    delete_flag = True
                    break
            if not delete_flag:
                filtered.append(u_doc)
        def delete_ids(ids):
            for id_ in ids:
                try:
                    transcribe_table.delete(f'id == "{id_}"')
                    print(f"Deleted {id_} from transcribe_table")
                except Exception as e:
                    print(f"Failed to delete {id_}: {e}")
        if ids_to_delete:
            threading.Thread(target=delete_ids, args=(ids_to_delete,)).start()
        return filtered

    def format_user_records(records):
        formatted = []
        for doc in records:
            content = doc.get("context", "")
            meta_data = {k: v for k, v in doc.items() if k not in ("context", "embedding")}
            doc_dict = {
                "content": content,
                "meta_data": meta_data,
            }
            formatted.append(json.dumps(doc_dict, indent=2))
        return "\n\n".join(formatted)

    filtered_user = filter_user_chunks(top_user, top_discarded, discard_threshold= discard_threshold)  
    user_context = format_user_records(filtered_user)
    system_msg = f"""You are an experienced podcast host with 15 years of experience conducting in-depth interviews across various industries. You have a natural curiosity about people's professional journeys and personal passions, and you're known for asking thought-provoking questions that reveal deeper insights about your guests.
    Your questioning style follows these core principles:{question_tone} 
   Generate exactly one question per response that embodies this interviewing persona and style."""
    final_prompt = system_msg + "\n\n" + query
    print("DEBUG - Query:", query)
    print("DEBUG - top_user:", len(top_user) if top_user else 0)
    print("DEBUG - top_discarded:", len(top_discarded) if top_discarded else 0)
    print("DEBUG - filtered_user:", len(filtered_user) if filtered_user else 0)
    print("DEBUG - user_context length:", len(user_context))
    print("DEBUG - user_context:", user_context[:200] + "..." if len(user_context) > 200 else user_context)

    # If no user context found from database search, use the original query directly
    #if user_context == "":
    #    print("DEBUG - No user context found, using direct query")
    #    direct_system_msg = f"All the generated questions are being asked in a podcast. Give only deep logical very good a single question based on the following content to ask guest. Just give question not anything else\n"
    #    direct_final_prompt = direct_system_msg + "\n\n" + query
    #    run_resp = agent.run(       
    #        direct_final_prompt,
    #        stream=False,
    #        search_knowledge=False,
    #        user_id= user_id
    #    )
    #    return run_resp.content.strip() if run_resp and run_resp.content else ""
    if user_context != "":
        run_resp = agent.run(       
            final_prompt,
            stream=False,
            search_knowledge=False,
            user_id= user_id
        )  
        return run_resp.content.strip() if run_resp and run_resp.content else ""



@app.post("/generate_question")
def generate_question_with_authority(req: TranscriptionRequest):
    # Split transcription text by dollar sign to separate paragraphs
    cleaned_transcription = req.transcription_text.replace("'","")
    raw_paragraphs = [p.strip() for p in cleaned_transcription.split('$') if p.strip()]
    
    # If no dollar signs found, split by periods as fallback
    if len(raw_paragraphs) <= 1:
        raw_paragraphs = [p.strip() for p in req.transcription_text.split('.') if p.strip()]
        # Group sentences into paragraphs
        paragraphs = []
        current_paragraph = ""
        
        for sentence in raw_paragraphs:
            current_paragraph += sentence + ". "
            if current_paragraph.strip():
                paragraphs.append(current_paragraph.strip())
                current_paragraph = ""
        
        # Add any remaining content
        if current_paragraph.strip():
            paragraphs.append(current_paragraph.strip())
    else:
        # Use the paragraphs as they are (from dollar sign splitting)
        paragraphs = raw_paragraphs
    
    docs = []
    for i, paragraph in enumerate(paragraphs):
        if not paragraph:
            continue
            
        context = paragraph.lower()
        emb = embedder.get_embedding(context)
        if not emb or len(emb) != 1536:
            continue  # skip or handle error
            
        doc_id = str(uuid.uuid4())
        docs.append({
            "id": str(doc_id),
            "context": context,
            "source": "user_info",
            "embedding": emb
        })
    

    if docs:
        transcribe_table.add(docs)
        
        # Analyze similarity scores using network analysis
        if len(docs) > 1:
            # Extract embeddings and texts
            embeddings = [doc["embedding"] for doc in docs]
            texts = [doc["context"] for doc in docs]
            
            # Build similarity matrix
            n = len(embeddings)
            sim_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        sim_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
                    else:
                        sim_matrix[i][j] = 1.0  # Self-similarity
            
            # Build similarity graph
            G = nx.from_numpy_array(sim_matrix)
            
            # Compute HITS scores
            try:
                hubs, authorities = nx.hits(G, max_iter=100, tol=1e-8)
                
                # Normalize authority scores to [0, 1]
                max_auth = max(authorities.values()) if authorities else 1.0
                norm_authorities = {k: v / max_auth for k, v in authorities.items()}
                
                # Print normalized scores
                print(f"Normalized Authority Scores (0-1), Top_k {req.top_k}:")
                for i, score in sorted(norm_authorities.items(), key=lambda x: x[1], reverse=True):
                    text_preview = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
                    print(f"{score:.4f} → {text_preview}")
                
                # Find the top_k paragraphs with highest authority scores
                sorted_authorities = sorted(norm_authorities.items(), key=lambda x: x[1], reverse=True)
                top_k = min(req.top_k, len(sorted_authorities))  # Don't exceed available documents
                
                top_authority_indices = [idx for idx, score in sorted_authorities[:top_k]]
                top_authority_texts = [texts[idx] for idx in top_authority_indices]
                top_authority_scores = [norm_authorities[idx] for idx in top_authority_indices]
                
                # Combine top_k highest authority texts for RAG query with Statement prefixes
                combined_statements = []
                for i, text in enumerate(top_authority_texts, 1):
                    combined_statements.append(f"Statement {i} {text}")
                combined_authority_text = " ".join(combined_statements)
                print(f"DEBUG - Combined authority text length: {len(combined_authority_text)}")
                print(f"DEBUG - Combined authority text preview: {combined_authority_text[:200]}")
                
                # Generate question based on the top_k highest authority paragraphs
                question = rag_query(combined_authority_text)
                print(f"DEBUG - Generated question: '{question}'")
                
                # Return analysis results
                analysis_results = {
                    "authority_scores": norm_authorities,
                    "texts": texts,
                    "top_k": top_k,
                    "top_authority_indices": top_authority_indices,
                    "top_authority_scores": top_authority_scores,
                    "top_authority_texts": top_authority_texts,
                    "combined_authority_text": combined_authority_text
                }
                
            except Exception as e:
                print(f"Error in network analysis: {e}")
                analysis_results = {"error": str(e)}
                question = ""
        else:
            # Single document case - use it directly for question generation
            if len(docs) == 1:
                single_text = docs[0]["context"]
                print(f"DEBUG - Single document case, using: {single_text[:200]}...")
                question = rag_query(single_text)
                analysis_results = {
                    "message": "Single document - no network analysis needed",
                    "single_text": single_text,
                    "top_k": 1
                }
            else:
                analysis_results = {"message": "Need at least 1 document for analysis"}
                question = "Need more content for analysis"
    
    return {
        "added": len(docs), 
        "analysis": analysis_results if 'analysis_results' in locals() else None,
        "question": question if 'question' in locals() else ""
    }
    
@app.get("/show_user_lancedb")
def show_user_lancedb():
    df = transcribe_table.to_pandas()
    records = df.to_dict(orient="records")
    for rec in records:
        emb = rec.get("embedding")
        if hasattr(emb, "tolist"):
            rec["embedding"] = emb.tolist()
        elif isinstance(emb, np.ndarray):
            rec["embedding"] = emb.tolist()
    return records

@app.get("/show_skipped_lancedb")
def show_skipped_lancedb():
    df = discarded_table.to_pandas()
    records = df.to_dict(orient="records")
    for rec in records:
        emb = rec.get("embedding")
        if hasattr(emb, "tolist"):
            rec["embedding"] = emb.tolist()
        elif isinstance(emb, np.ndarray):
            rec["embedding"] = emb.tolist()
    return records

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_rag:app", host="0.0.0.0", port=8000, reload=True)
