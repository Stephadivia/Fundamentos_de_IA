#=================================================
# Graficas de aprendizaje
#=================================================
#Stephania Valdivia Diaz
#Fundamentos de IA
#=================================================
 
#==========================
# Modulos necesarios
#==========================
import matplotlib.pyplot as plt
from IPython import display

#plt.ion()

#======================================
# Grafica dinamica en el tiempo
#======================================
def graficar(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.ion()  # Modo interactivo
    plt.clf()
    plt.title('Entrenando...')
    plt.xlabel('Numero de juegos')
    plt.ylabel('Puntaje')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
