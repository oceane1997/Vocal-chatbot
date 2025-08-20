# 🗣️ Chatbot vocal (Candide) — Streamlit

Appli qui répond à des questions sur *Candide* :
- **Texte** : question tapée → réponse via similarité (Jaccard) sur le corpus.
- **Voix** : audio (WAV/FLAC) ou **micro navigateur** → transcription (Google Web Speech) → même chatbot.

## Fonctionnement
1) Chargement du corpus (`corpus.txt`) et prétraitement FR (tokenisation, stopwords, stemming).
2) Similarité **Jaccard** question↔phrases, avec **seuil** ajustable (slider).
3) Entrées possibles :
   - **Texte** (prioritaire si combiné)
   - **Fichier audio** (WAV/FLAC)
   - **Micro navigateur** (`st.audio_input`)
4) Messages d’erreur lisibles (inaudible, format non supporté, réseau…).

## Lancer en local
```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
