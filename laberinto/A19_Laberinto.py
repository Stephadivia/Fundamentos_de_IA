#=================================================
# Laberinto
#=================================================
#Stephania Valdivia Diaz
#Fundamentos de IA
#=================================================

#=================================================
# De pyamaze.py en el directorio laberinto
#=================================================
from laberinto.pyamaze import maze, agent, COLOR

#=================================================
# Crear laberinto
# x, y posicion de la meta
#=================================================
m = maze(25, 40)
m.CreateMaze(x=1, y=1)

#=================================================
# Poner agente en el laberinto
#=================================================
a=agent(m, footprints=True, filled=True)

#=================================================
# Graficar trayectoria del agente
#=================================================
m.tracePath({a:m.path}, delay=25)
m.run()