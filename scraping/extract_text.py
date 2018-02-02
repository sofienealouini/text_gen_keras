import json
import pandas as pd
import numpy as np
from processing import get_alphabet
import string


# Charger le fichiers de données
with open('speeches.json') as data_file:
    data = pd.DataFrame(json.load(data_file))


# Sélectionner les discours de candidats
candidats = data[data['fonction'].str.contains("candidat|Candidat|CANDIDAT")]
speeches = [speech.lower() for speech in list(candidats["discours"])]
corpus = " ".join(speeches)
print(len(corpus))


# Caractères à éliminer
def chars_to_del(text, target_chars):
    text_chars = get_alphabet(text)
    del_chars = list(set(text_chars) - set(target_chars))
    return del_chars


# Caractères qu'on veut avoir dans le texte final
spec_chars = [' ', '\'']
letters_fr = list(string.ascii_lowercase)
letters_acc = []
punct = ['.',',']
digits = []
#digits = list(string.digits)
target_chars = spec_chars + letters_fr + letters_acc + punct + digits
target_chars.sort()


# Caractères à remplacer
letters_replace = {'à':'a',
                   'è':'e',
                   'é':'e',
                   'ç':'c',
                   'â':'a',
                   'ê':'e',
                   'î':'i',
                   'ï':'i',
                   'ô':'o',
                   'û':'u',
                   'ü':'u',
                   'ù':'u',
                   '!':'.',
                   '?':'.'}


# Caractères à supprimer (remplacer par un espace)
del_chars = chars_to_del(corpus, target_chars + list(letters_replace.keys()))
letters_delete = {k:" " for k in del_chars}


# Nettoyage du texte (optimisation possible)
d = dict(letters_replace)
d.update(letters_delete)
for k in d:
    corpus = corpus.replace(k, d[k])
corpus = ' '.join(corpus.split())


# Exporter le fichier
file = open("candidats.txt", "w")
file.write(corpus)
file.close()