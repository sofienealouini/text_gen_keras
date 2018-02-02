from keras.models import Sequential, load_model
import numpy as np
import argparse

from params import PRED_LEN, SEQ_LEN
from data import FIRST_SENTENCE, CHAR_INDEX, INDEX_CHAR, NB_CHARS
from processing import encode


#np.random.seed(80)


# Sampling dans la distribution de probabilités prédite
def sample(preds, temperature=1.0, do_sample=True):
    # Avec sampling
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    # Sans sampling
    preds = np.reshape(preds, (1, preds.shape[0]))
    if do_sample:
        return np.argmax(probas)
    else:
        return np.argmax(preds)


def predict_single_input(model, sentence):
    x = encode(sentence, CHAR_INDEX, NB_CHARS, float)
    x = np.reshape(x, (1, x.shape[0], x.shape[1]))
    preds = model.predict(x, verbose=0)[0]
    if len(preds.shape) > 1:
        return preds[-1]
    else:
        return preds


# Génération d'un paragraphe commençant par FIRST_SEQUENCE
def predict_paragraph(pred_len, model, do_sample=True, temperature=1.0):
    first_s = FIRST_SENTENCE
    generated = first_s
    for i in range(pred_len):
        preds = predict_single_input(model, first_s)
        next_index = sample(preds, temperature, do_sample)
        next_char = INDEX_CHAR[next_index]
        generated += next_char
        first_s = first_s[1:] + next_char
    return generated


# Récupération des activations (valeurs) d'une couche donnée
def predict_activations(complete_model, predicted_paragraph, tar_layer):
    activations = []
    # Création du modèle tronqué
    partial_model = Sequential()
    for l in range(tar_layer):
        partial_model.add(complete_model.layers[l])
    partial_model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Récupération des outputs de la couche à chaque caractère prédit
    for i in range(len(predicted_paragraph)-SEQ_LEN):
        sentence = predicted_paragraph[i:i+SEQ_LEN]
        act = predict_single_input(partial_model, sentence)
        activations.append(act)
    activations = np.transpose(activations)
    fs_l_acts = [[0] * SEQ_LEN + list(a) for a in activations]
    return np.array(fs_l_acts)


# Récupération de toutes les activations du réseau
def all_activations(complete_model, predicted_paragraph):
    net_acts = {}
    for layer in range(1, 1 + len(MODEL.layers)):
        net_acts[layer] = predict_activations(MODEL, predicted_paragraph, layer)
    return net_acts


# Arguments à fournir dans le terminal
parser = argparse.ArgumentParser(description='generates text from seed based on a trained model')
parser.add_argument('model', help='model used to generate text')
parser.add_argument('-t', '--temperature', help='temperature for sampling', type=float)
args = parser.parse_args()


# Charger le modèle à utiliser
MODEL = load_model(args.model)


# Fixer la température pour le sampling
if args.temperature is not None:
    TEMP = args.temperature
else:
    TEMP = 0.2


# Prediction
paragraph = predict_paragraph(PRED_LEN, MODEL, True, TEMP)
print(paragraph)


'''
# Prédire plusieurs paragraphes
pars = []
for k in range(50):
    np.random.seed(k ** 2)
    pars.append(predict_paragraph(PRED_LEN, MODEL, True, TEMP))

with open("example/pars.txt", "w") as f:
    for par in pars:
        f.write(par + "\n \n")



# Activations
net_acts = all_activations(MODEL, paragraph)
np.save("example/activations.npy", net_acts)'''
