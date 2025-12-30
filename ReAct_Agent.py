import os
import re
import json
import numpy as np
import requests
import string
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# --- 1. AYARLAR VE MODEL YÜKLEMELERİ ---
# (Eski kodunla aynı, burası veri tabanı ve modelleri hazırlar)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3" 
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

# Dosya yolları (Senin dosya isimlerin)
EMBED_FILE = "tbk_chunks_embeddings.npy"
META_FILE = "tbk_chunks_metadata.json"

print("🔄 Modeller ve veriler yükleniyor...")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)

# Verileri yükle
embeddings = np.load(EMBED_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# BM25 Hazırlığı
def simple_tokenizer(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

tokenized_corpus = [simple_tokenizer(doc["text_preview"]) for doc in metadata]
bm25 = BM25Okapi(tokenized_corpus)

print("✅ Sistem ve Veri Tabanı Hazır!\n")


# --- 2. TEMEL ARAMA FONKSİYONLARI (ESKİ KODUN) ---
# Bu fonksiyonlar "Motor" kısmıdır. ReAct ajanı bunları kullanacak.

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))

def search_semantic(query, top_k=10):
    q_emb = embed_model.encode(query, convert_to_numpy=True)
    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(q_emb, emb)
        scores.append((i, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def search_bm25(query, top_k=10):
    tokenized_query = simple_tokenizer(query)
    doc_scores = bm25.get_scores(tokenized_query)
    scores = [(i, score) for i, score in enumerate(doc_scores)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def hybrid_search_and_rerank(query, top_k_final=3):
    # 1. Semantic ve BM25 ile adayları bul
    semantic_results = search_semantic(query, top_k=10)
    bm25_results = search_bm25(query, top_k=10)
    
    unique_indices = set([i for i, _ in semantic_results] + [i for i, _ in bm25_results])
    candidate_indices = list(unique_indices)
    
    if not candidate_indices:
        return []

    # 2. Rerank (Yeniden Puanlama)
    cross_inp = [[query, metadata[idx]["text_preview"]] for idx in candidate_indices]
    rerank_scores = reranker_model.predict(cross_inp)
    
    final_results = []
    for i, idx in enumerate(candidate_indices):
        final_results.append({
            "chunk_id": idx,
            "score": float(rerank_scores[i]),
            "text": metadata[idx]["text_preview"]
        })
    
    return sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k_final]


# --- 3. TOOL (ARAÇ) TANIMI (PROJE GEREKLİLİĞİ ADIM 3.1) ---
# Ajanın kullanacağı "Tool" fonksiyonu.
# ÖNEMLİ: Bu fonksiyon LLM cevabı döndürmez, ham bilgi (Observation) döndürür.

def kira_mevzuati_ara_tool(sorgu_metni):
    """
    Kira hukuku ve mevzuatı hakkında arama yapar.
    Girdi: Sorgu metni (string)
    Çıktı: Bulunan döküman metinleri (string)
    """
    print(f"\n🔎 [TOOL ÇALIŞIYOR] Sorgu: {sorgu_metni}")
    results = hybrid_search_and_rerank(sorgu_metni, top_k_final=3)
    
    if not results:
        return "Aranan konuda veritabanında bilgi bulunamadı."
    
    observation_text = ""
    for r in results:
        observation_text += f"---\n[Döküman Parçası]\n{r['text']}\n"
        
    return observation_text


# --- 4. REACT AJAN MİMARİSİ (PROJE GEREKLİLİĞİ ADIM 3.2) ---
# Burası "Beyin" kısmıdır.

SYSTEM_PROMPT = """
Sen uzman bir Kira Hukuku Asistanısın. Görevin sorulan sorulara net cevap vermektir.

ELİNDEKİ ARAÇLAR:
1. kira_mevzuati_ara: Kira kanunu ile ilgili bilgi arar.

TAKİP ETMEN GEREKEN FORMAT:
Soru: Kullanıcının sorusu
Thought: Cevabı biliyor muyum? Bilmiyorsam hangi aracı kullanmalıyım?
Action: kira_mevzuati_ara: "aranacak kelimeler"
Observation: (Buraya arama sonucu gelecek)
Thought: Gelen bilgiyi okudum. Cevap bu metinde var mı? Varsa Final Answer yaz.
Final Answer: Sorunun cevabı (Türkçe).

ÇOK ÖNEMLİ KURALLAR:
1. EĞER "Observation" kısmında bilgi görüyorsan, TEKRAR ARAMA YAPMA. Hemen "Final Answer" yaz.
2. "Action:" yazarken sadece `kira_mevzuati_ara: "kelime"` formatını kullan. Başka bir şey yazma.
3. Asla kendi kendine Observation uydurma.
"""

def ask_ollama(prompt):
    """Ollama API'sine istek atar."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "stop": ["Observation:"]} # Observation'ı modelin uydurmasını engelle
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return response.json()["response"]
    except Exception as e:
        return f"Hata: {e}"

def react_loop(user_question):
    conversation_history = f"{SYSTEM_PROMPT}\n\nSoru: {user_question}\n"
    max_turns = 5
    turn_count = 0
    
    # Modelin daha önce yaptığı aramaları hafızada tutalım (Loop engellemek için)
    previous_actions = []

    print(f"\n🤖 [AJAN] '{user_question}' sorusu üzerine düşünmeye başladı...\n")

    while turn_count < max_turns:
        turn_count += 1
        
        response = ask_ollama(conversation_history)
        response = response.strip()
        print(f"\n--- Adım {turn_count} ---")
        print(response)
        
        conversation_history += f"{response}\n"

        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()

        # Action Yakalama
        action_match = re.search(r"Action:\s*(\w+):\s*\"?([^\"]+)\"?", response)
        
        if action_match:
            tool_name = action_match.group(1)
            query = action_match.group(2).strip()
            
            # --- YENİ EKLENEN GÜVENLİK ÖNLEMİ ---
            # Eğer bu aramayı daha önce yaptıysa engelle!
            if query in previous_actions:
                observation = "UYARI: Bu aramayı zaten yaptın ve yukarıda sonuçları var. Tekrar arama yapma! Yukarıdaki metni oku ve 'Final Answer' ver."
                print(f"⚠️ [LOOP ENGELENDİ] Model aynı şeyi ({query}) tekrar aramak istedi.")
            else:
                # Yeni bir arama ise çalıştır
                if tool_name == "kira_mevzuati_ara":
                    observation = kira_mevzuati_ara_tool(query)
                    previous_actions.append(query) # Listeye ekle
                else:
                    observation = f"Hata: {tool_name} diye bir araç yok. Sadece 'kira_mevzuati_ara' kullanabilirsin."
            
            observation_str = f"Observation: {observation}\n"
            conversation_history += observation_str
            
        else:
            # Action yoksa ve Final Answer da yoksa, model saçmalamış olabilir.
            # Ona zorla cevap vermesini söyleyelim.
            if turn_count == max_turns:
                 return "Üzgünüm, döngüye girdim. Lütfen soruyu tekrar sor."
            
    return "Maksimum adım sayısına ulaşıldı (Cevap bulunamadı)."


# --- 5. ANA ÇALIŞTIRMA BLOĞU ---
if __name__ == "__main__":
    while True:
        print("\n" + "="*60)
        q = input("Soru Sor (Çıkış için 'q'): ").strip()
        if q.lower() == 'q': break
        
        final_response = react_loop(q)
        
        print("\n🎯 [SONUÇ]:")
        print(final_response)