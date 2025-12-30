# 🏛️ Rental Law ReAct Agent (Turkish)

An autonomous AI agent designed to answer questions about **Turkish Rental Law (Türk Borçlar Kanunu)**. This project transforms a static LLM into a reasoning engine using the **ReAct (Reasoning + Acting)** architecture.

Instead of hallucinating answers, this agent uses a **Knowledge Base** as a tool, performs **Hybrid Search (Semantic + Keyword)**, and iteratively reasons to provide accurate legal information.

---

## 🚀 Key Features

* **Architecture:** ReAct (Reasoning, Acting, Observation) loop[cite: 59].
* **Method:** RAG (Retrieval-Augmented Generation) utilized as a *Tool* rather than a static pipeline[cite: 84].
* **Search Engine:** Advanced Hybrid Search combining **BM25** (Keyword) and **Cosine Similarity** (Semantic), refined by a **Cross-Encoder Reranker**.
* **Local LLM:** Powered by **Llama 3.2 (3B)** running locally via Ollama.
* **Safety:** Includes loop protection (max_turns) to prevent infinite reasoning cycles[cite: 52].

## 🛠️ Architecture Overview

[cite_start]The system consists of three main layers[cite: 91]:

1.  **The Brain (Orchestrator):** Llama 3.2 model that decides *when* to search and *how* to answer.
2.  **The Tools (Limbs):** A custom Python function `kira_mevzuati_ara` that queries the vector database.
3.  **The Knowledge (Memory):** Chunks of the Turkish Code of Obligations (TBK) embedded with `paraphrase-multilingual-MiniLM-L12-v2`.

---

## 💻 Installation & Setup

### Prerequisites
* Python 3.8+
* [Ollama](https://ollama.com/) installed and running.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Rental-Law-ReAct-Agent.git](https://github.com/YOUR_USERNAME/Rental-Law-ReAct-Agent.git)
cd Rental-Law-ReAct-Agent 
```

### 2. Install Dependencies
``` Bash
pip install -r requirements.txt
```
### 3. Pull the Local Model
Ensure Ollama is running and pull the lightweight Llama 3.2 model:

```Bash
ollama pull llama3.2:3b
``` 
### 4. Run the Agent
```Bash
python ReAct_Agent.py
```
## 📝 Usage & Trace Examples
The agent follows the Thought -> Action -> Observation pattern.

### Scenario A: One-Shot Query (Direct Answer) 

User: "Kiracı olarak hangi haklara sahibim?" (What rights do I have as a tenant?)

```Plaintext

--- Step 1 ---
Thought: Do I know the answer? If not, which tool should I use?
Action: kira_mevzuati_ara: "kiracı hakları"

 🔎 [TOOL RUNNING] Query: kiracı hakları
 👀 [OBSERVATION] Retrieved data from database (Length: 969 chars)... 
(Contains TBK articles about sub-leasing and usage rights)

--- Step 2 ---
Thought: I have read the information. Is the answer in this text? Yes.
Final Answer: Kiracı olarak aşağıdaki haklara sahiptir:
- Kiraya verene zarar verecek bir değişikliğe yol açmamak koşuluyla...
- Kullanım hakkını da başkasına devredebilir...
```
### Scenario B: Reasoning Loop
The agent is designed to self-correct. If it repeats an action or gets stuck, the system intervenes to force a final answer based on available context.

### 📂 Project Structure
* **ReAct_Agent.py:** Main application code containing the ReAct loop and RAG tools.
* **tbk_chunks_metadata.json:** The knowledge base (Legal text chunks).
* **tbk_chunks_embeddings.npy:** Pre-computed vector embeddings for fast search.
* **requirements.txt:** List of Python dependencies.

### ⚠️ Notes
* This project runs entirely offline (Locally).
* The model used (3B parameters) is lightweight; prompt engineering techniques were used to ensure strict adherence to the ReAct format.

## Developer: Burak Dağdeviren
