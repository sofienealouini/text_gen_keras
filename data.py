from params import LOCAL_PATH, SEQ_LEN, SEQ_SKIP, PRED_START_INDEX
from processing import get_alphabet, create_dicts, encode, create_data


# Texte initial
TEXT = open(LOCAL_PATH).read().lower()[0:1500000]

# Ensemble des caractères distincts du texte
ALPHABET = get_alphabet(TEXT)

# Nombre de caractères distincts
NB_CHARS = len(ALPHABET)

# Dictionnaires utilisés pour l'encodage et le décodage
CHAR_INDEX, INDEX_CHAR = create_dicts(ALPHABET)

# Texte encodé (one-hot encoding de chaque caractère du texte)
ENCODED_TEXT = encode(TEXT, CHAR_INDEX, NB_CHARS, bool)

# Données d'entraînement
# Un input est une séquences de caractères encodés
# Un target est le caractère suivant cette séquence (la valeur à prédire)
INPUTS, TARGETS = create_data(ENCODED_TEXT, SEQ_LEN, SEQ_SKIP)

# Phrase de départ de la prédiction
#FIRST_SENTENCE = TEXT[PRED_START_INDEX:PRED_START_INDEX+SEQ_LEN]
FIRST_SENTENCE = "mes chers compatriotes nous abordons en ce moment "