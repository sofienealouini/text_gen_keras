import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from processing import to_array


def draw(texte, activations, line_length=50):

    # Data preparation
    nb_layers = len(activations.keys())
    text_array = to_array(texte, line_length)
    total_cells = activations[1].shape[0]

    # Default plot (layer 1, neuron 0)
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    norm = Normalize(-1.0, 1.0)
    act = activations[1][0,:]
    act_array = to_array(act, line_length)
    tab = ax.table(cellText = text_array, loc='center', colWidths=[0.016]*(line_length),
                   cellLoc='center', cellColours=plt.cm.RdYlGn(norm(act_array)))

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    
    l0 = 1
    axlayer = plt.axes([0.2, 0.2, 0.63, 0.03], facecolor=axcolor)
    slayer = Slider(axlayer, 'Layer', 1, nb_layers, valinit=l0, valfmt='%0.0f')
    
    n0 = 0
    axneuron = plt.axes([0.2, 0.15, 0.63, 0.03], facecolor=axcolor)
    sneuron = Slider(axneuron, 'Neuron', 0, total_cells, valinit=n0, valfmt='%0.0f')

    def upd_neuron_change(val):
        layer = int(round(slayer.val))
        neuron = int(round(sneuron.val))
        act = activations[layer][neuron, :]
        act_array = to_array(act, line_length)
        for k in tab._cells.keys():
            tab._cells[k].set_fc(plt.cm.RdYlGn(norm(act_array[k])))

    sneuron.on_changed(upd_neuron_change)

    def upd_layer_change(val):
        layer = int(round(slayer.val))
        neuron = int(round(sneuron.val))
        if sneuron.val > activations[layer].shape[0]:
            sneuron.set_val(0)
            neuron = int(0)
        sneuron.valmax = activations[layer].shape[0]
        act = activations[layer][neuron,:]
        act_array = to_array(act, line_length)
        for k in tab._cells.keys():
            tab._cells[k].set_fc(plt.cm.RdYlGn(norm(act_array[k])))

    slayer.on_changed(upd_layer_change)
    plt.show()



activations = np.load("example/activations.npy").item()
with open("example/predicted.txt", "r") as f:
    texte = f.read()

draw(texte,activations)