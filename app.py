# app_voice_chatbot.py — Chatbot vocal (Candide) : texte OU voix (micro navigateur / fichier)

import streamlit as st
import nltk
from typing import Optional
from pathlib import Path
import tempfile
import speech_recognition as sr

# ============================
# Étape 1 — Imports & setup
# ============================
st.set_page_config(page_title="Chatbot vocal", page_icon="🗣️")
st.title("🗣️ Chatbot vocal (Candide)")

# Téléchargements NLTK (inoffensifs si déjà présents)
for pkg in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

st.success("Imports OK : Streamlit, SpeechRecognition, NLTK")

# ============================
# Étape 2 — Chatbot : corpus + prétraitement
# ============================
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

STOPS_FR = set(stopwords.words("french"))
STEM_FR = SnowballStemmer("french")

def preprocess(sentence: str) -> list[str]:
    """Tokenisation FR, minuscules, lettres uniquement, stopwords, stemming."""
    toks = word_tokenize(sentence, language="french")
    toks = [t.lower() for t in toks if t.isalpha()]
    toks = [t for t in toks if t not in STOPS_FR]
    toks = [STEM_FR.stem(t) for t in toks]
    return toks

@st.cache_data
def load_corpus(path: str = "corpus.txt"):
    p = Path(path)
    if not p.exists():
        return None, None, f"❗️ Fichier {path} introuvable. Place-le à côté de app_voice_chatbot.py."
    text = p.read_text(encoding="utf-8").replace("\n", " ")
    sentences = sent_tokenize(text, language="french")
    return text, sentences, None

@st.cache_data
def build_processed(sentences: list[str]):
    """Crée les ensembles de tokens (pour similarité Jaccard)."""
    return [set(preprocess(s)) for s in sentences]

st.header("Étape 2/6 — Corpus & prétraitement")
full_text, sentences, err = load_corpus("corpus.txt")
if err:
    st.error(err)
    st.stop()

corpus_processed = build_processed(sentences)
st.success(f"Corpus chargé ✅ — {len(sentences)} phrases.")
st.caption("Prétraitement OK (stopwords FR + stemming).")

# Stockage robuste pour réutiliser même après reruns Streamlit
if "sentences" not in st.session_state:
    st.session_state["sentences"] = sentences
if "corpus_processed" not in st.session_state:
    st.session_state["corpus_processed"] = corpus_processed

# ============================
# Étape 3 — Transcription (fichier / blob micro navigateur)
# ============================
def transcribe_audio_file(uploaded_file, language: str = "fr-FR"):
    """
    Transcrit un fichier audio/flux (WAV/FLAC/WEBM) avec Google Web Speech API.
    uploaded_file : st.file_uploader ou st.audio_input (a un .read()).
    Renvoie (message_utilisateur, debug_details_ou_None)
    """
    r = sr.Recognizer()

    # Écrire l'upload dans un fichier temporaire avec extension
    suffix = ".wav"
    try:
        name = getattr(uploaded_file, "name", None)
        if name and "." in name:
            suffix = "." + name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)
    except Exception as e:
        return ("❗️ Impossible de créer un fichier temporaire pour l'audio.", f"{type(e).__name__}: {e}")

    try:
        with sr.AudioFile(str(tmp_path)) as source:
            # léger ajustement du bruit
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.record(source)
        text = r.recognize_google(audio, language=language)
        return (text, None)

    except sr.UnknownValueError as e:
        return ("❗️ L’API n’a pas compris l’audio. Parlez plus clairement ou vérifiez la langue.", f"{type(e).__name__}")
    except sr.RequestError as e:
        return ("❗️ Erreur de service (réseau/quota). Vérifiez Internet et réessayez.", f"{type(e).__name__}: {e}")
    except ValueError as e:
        return ("❗️ Format non supporté. Utilisez un fichier WAV ou FLAC.", f"{type(e).__name__}: {e}")
    except Exception as e:
        return ("❗️ Erreur inattendue lors de la transcription.", f"{type(e).__name__}: {e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# (facultatif) petit test de la transcription seule
#with st.expander("🧪 Étape 3/6 — Tester la transcription (fichier)"):
    lang_test = st.selectbox("Langue audio (test)", ["fr-FR", "en-US", "es-ES"], index=0, key="lang_test")
    test_audio = st.file_uploader("Choisir un audio (WAV/FLAC)", type=["wav", "flac"], key="test_audio")
    if st.button("Tester la transcription", key="btn_test_trans"):
        if not test_audio:
            st.warning("Sélectionnez un fichier audio d'abord.")
        else:
            with st.spinner("Transcription en cours..."):
                msg, dbg = transcribe_audio_file(test_audio, language=lang_test)
            if msg.startswith("❗️"):
                st.error(msg)
                if dbg:
                    st.code(dbg)
            else:
                st.success("Transcription OK")
                st.write(msg)

# ============================
# Étape 4 — Chatbot (Jaccard) + routeur texte/audio
# ============================
def get_most_relevant_sentence(query: str, sentences: list[str], corpus_processed: list[set]) -> tuple[str, float]:
    """Renvoie (phrase_la_plus_pertinente, score_jaccard)."""
    q = set(preprocess(query))
    if not q:
        return ("(Votre question est vide après prétraitement.)", 0.0)

    best_idx, best_score = -1, 0.0
    for i, s_tokens in enumerate(corpus_processed):
        inter = q & s_tokens
        union = q | s_tokens
        score = (len(inter) / len(union)) if union else 0.0
        if score > best_score:
            best_score, best_idx = score, i

    if best_idx == -1:
        return ("(Aucune phrase trouvée.)", 0.0)
    return (sentences[best_idx], best_score)

def chatbot(question: str, sentences: list[str], corpus_processed: list[set], min_score: float = 0.15) -> str:
    """Si le score Jaccard < min_score → on demande de reformuler."""
    best_sentence, score = get_most_relevant_sentence(question, sentences, corpus_processed)
    if score < min_score:
        return "Je n’ai pas trouvé de passage assez proche. Peux-tu reformuler ta question ?"
    return best_sentence

def reply_from_text_or_audio(
    text_input: Optional[str],
    audio_file,
    language_code: str,
    sentences: list[str],
    corpus_processed: list[set],
    min_score: float = 0.15,
):
    """
    - Si text_input est non vide → chatbot(question).
    - Sinon, si audio_file est fourni → transcription puis chatbot(transcription).
    - Renvoie (réponse, debug) ; debug != None seulement si erreur de transcription.
    """
    if text_input and text_input.strip():
        question = text_input.strip()
        answer = chatbot(question, sentences, corpus_processed, min_score=min_score)
        return (answer, None)

    if audio_file is not None:
        msg, dbg = transcribe_audio_file(audio_file, language=language_code)
        if msg.startswith("❗️"):
            return (msg, dbg)
        answer = chatbot(msg, sentences, corpus_processed, min_score=min_score)
        return (answer, None)

    return ("❗️ Fournis soit une question texte, soit un fichier audio.", None)

# ============================
# Étape 5 — Interface : Texte OU Audio + Micro navigateur
# ============================
st.header("Étape 5/6 — Pose ta question (texte OU audio)")

# Paramètres
colA, colB = st.columns([1, 1])
with colA:
    lang = st.selectbox("Langue de l'audio", ["fr-FR", "en-US", "es-ES"], index=0)
with colB:
    min_score = st.slider("Seuil de pertinence (Jaccard)", 0.0, 1.0, 0.15, 0.01,
                          help="Si le score < seuil → on demande de reformuler.")

# Entrées possibles
txt_question = st.text_input("Question (texte)")
audio_file = st.file_uploader("OU dépose un fichier audio (WAV/FLAC)", type=["wav", "flac"])

if txt_question and audio_file:
    st.info("Texte **prioritaire** : si vous fournissez les deux, le texte est utilisé et l’audio est ignoré.")

# Bouton principal
if st.button("Répondre", key="btn_text_or_file"):
    with st.spinner("Analyse en cours..."):
        answer, debug = reply_from_text_or_audio(
            text_input=txt_question,
            audio_file=audio_file,
            language_code=lang,
            sentences=st.session_state["sentences"],
            corpus_processed=st.session_state["corpus_processed"],
            min_score=min_score,
        )

    st.markdown("**Réponse du chatbot :**")
    if answer.startswith("❗️"):
        st.error(answer)
        if debug:
            with st.expander("Détails techniques"):
                st.code(debug)
    else:
        st.success("Réponse trouvée ✅")
        st.write(answer)

# Micro (navigateur) — pas de PyAudio requis
st.markdown("---")
st.subheader("🎙️ Micro (navigateur) — poser une question à voix haute")

blob = st.audio_input("Appuie pour enregistrer puis relâche")

if st.button("Répondre (depuis le micro)", key="btn_mic"):
    if not blob:
        st.warning("Enregistre d’abord un segment au micro.")
    else:
        with st.spinner("Transcription en cours..."):
            msg, dbg = transcribe_audio_file(blob, language=lang)

        if msg.startswith("❗️"):
            st.error(msg)
            if dbg:
                with st.expander("Détails techniques"):
                    st.code(dbg)
        else:
            st.caption("**Transcription détectée :**")
            st.write(msg)

            answer = chatbot(
                msg,
                st.session_state["sentences"],
                st.session_state["corpus_processed"],
                min_score=min_score
            )
            st.markdown("**Réponse du chatbot :**")
            st.success("Réponse trouvée ✅")
            st.write(answer)
