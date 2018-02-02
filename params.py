# Paramètres de prétraitement
LOCAL_PATH = "candidats.txt"
SEQ_LEN = 50
SEQ_SKIP = 3

# Paramètres d'apprentissage
NB_ITER = 50
BATCH_SIZE = 64

# Paramètres de la prédiction (longueur du paragraphe à prédire, séquence de départ)
PRED_LEN = 500
PRED_START_INDEX = 1500

# Paramètres visualisation
TAR_LAYER = 6   # 1-indexed
TAR_CELL = 0    # 0-indexed