# preprocessing_masker.py (Version 2)

import json
from pathlib import Path
import spacy
import nltk
from nltk.corpus import opinion_lexicon
from llm import call_ai
from logging_config import logger
from typing import Union
from pathlib import Path
import re
# --- Konfiguration ---
# --- Konfiguration ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None

HIGH_PRIORITY_LEXICON = {
    "stupid", "idiot", "moron", "clown", "liar", "scumbag", "traitor", "criminal",
    "disgraceful", "shameful", "lunatic", "awful", "terrible", "horrible", "worst",
    "pos", "warrior", "hero", "villain", "disgusting", "pathetic", "failed", "conned"
}

try:
    NLTK_LEXICON = set(opinion_lexicon.words())
    logger.info(f"NLTK opinion lexicon loaded with {len(NLTK_LEXICON)} words.")
except LookupError:
    logger.error("NLTK opinion_lexicon not found. Run: import nltk; nltk.download('opinion_lexicon')")
    NLTK_LEXICON = set()

URL_REGEX = re.compile(r"^\s*https?://[^\s]+\s*$")

# --- Kernlogik der Maskierung ---

def create_masked_text(text: str) -> str:
    """
    Maskiert ALLE als meinungstragend identifizierten Wörter durch eine Kombination
    aus deterministischen Regeln und LLM-Analyse.
    """
    if not text or not nlp:
        return "[MASKED]"
    if URL_REGEX.match(text):
        return "[MASKED]" # Reine URLs werden zu einem einzelnen Mask-Token

    doc = nlp(text)
    words = [token.text for token in doc]
    
    # --- Schritt 1: Deterministische Analyse ---
    deterministic_indices = set()
    word_types_to_check = {"ADJ", "ADV", "VERB", "NOUN"}
    for token in doc:
        if token.lower_ in HIGH_PRIORITY_LEXICON:
            deterministic_indices.add(token.i)
        elif token.pos_ in word_types_to_check and token.lower_ in NLTK_LEXICON:
            deterministic_indices.add(token.i)
    
    logger.debug(f"Deterministisch gefundene Indizes: {deterministic_indices}")

    # --- Schritt 2: LLM-Analyse ---
    llm_indices = set()
    try:
        # Der Prompt in masking.py/templates.py sollte bereits "maskiere alle Wörter" anweisen
        from masking import mask_text as mask_text_llm
        llm_masked_text = mask_text_llm(text)

        # Vergleiche Original mit LLM-Antwort, um maskierte Indizes zu finden
        llm_words = llm_masked_text.split()
        if len(words) == len(llm_words): # Nur vergleichen, wenn die Längen übereinstimmen
            for i, word in enumerate(llm_words):
                if word == "[MASKED]" and words[i] != "[MASKED]":
                    llm_indices.add(i)
        logger.debug(f"LLM-gefundene Indizes: {llm_indices}")

    except Exception as e:
        logger.error(f"LLM-Aufruf für 'Union'-Strategie fehlgeschlagen: {e}")

    # --- Schritt 3: Kombination (Die "Union") ---
    final_indices_to_mask = deterministic_indices.union(llm_indices)
    logger.debug(f"Finale Indizes zum Maskieren: {final_indices_to_mask}")

    # --- Schritt 4: Finale Maskierung ---
    if final_indices_to_mask:
        for i in final_indices_to_mask:
            # Stelle sicher, dass der Index gültig ist
            if i < len(words):
                words[i] = "[MASKED]"
        return " ".join(words)

    # --- Schritt 5: Sicherheitsnetz ---
    # Nur wenn BEIDE Methoden nichts finden, greift der Failsafe
    logger.warning("Weder Regeln noch LLM haben etwas zum Maskieren gefunden. Aktiviere Failsafe.")
    failsafe_priority = ["ADJ", "ADV", "VERB", "NOUN"]
    for pos_tag in failsafe_priority:
        for token in reversed(doc):
            if token.pos_ == pos_tag:
                words[token.i] = "[MASKED]"
                return " ".join(words)
    
    # Letzte Rettung
    if words:
        words[-1] = "[MASKED]"
    return " ".join(words)



# --- Datei-Verarbeitung (bleibt identisch) ---
def process_user_file(file_path: Union[str, Path]):
    # ... (Dieser Teil bleibt unverändert)
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Datei nicht gefunden: {file_path}")
        return
    logger.info(f"Beginne Pre-Processing für: {file_path.name}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) < 2:
        logger.warning(f"Datei {file_path.name} hat weniger als 2 Zeilen. Überspringe Maskierung.")
        return
    try:
        holdout_data = json.loads(lines[1])
        if "tweets" not in holdout_data:
            logger.warning(f"'tweets' nicht in Zeile 2 von {file_path.name} gefunden.")
            return
        for tweet in holdout_data.get("tweets", []):
            if "full_text" in tweet and "masked_text" not in tweet:
                original_text = tweet["full_text"]
                masked_version = create_masked_text(original_text)
                tweet["masked_text"] = masked_version
                logger.debug(f"Tweet {tweet.get('tweet_id', '')} maskiert.")
        lines[1] = json.dumps(holdout_data, ensure_ascii=False) + '\n'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        logger.info(f"Pre-Processing für {file_path.name} erfolgreich abgeschlossen.")
    except json.JSONDecodeError:
        logger.error(f"Fehler beim Parsen von JSON in Zeile 2 von {file_path.name}.")
    except Exception as e:
        logger.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

if __name__ == '__main__':
    test_file = Path("data/test_user/4252893976.0.jsonl") 
    if test_file.exists():
        process_user_file(test_file)
        with open(test_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        print("\n--- Ergebnis in Zeile 2: ---")
        modified_holdout = json.loads(all_lines[1])
        for i, t in enumerate(modified_holdout.get("tweets", [])[:3]):
             print(f"\nTweet {i+1}:")
             print(f"  Original: {t.get('full_text')}")
             print(f"  Maskiert: {t.get('masked_text')}")
    else:
        print(f"Testdatei {test_file} nicht gefunden.")