#=================
# Clase particle
#=================
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, n_clusters, data, use_kmeans=True, w=0.72, c1=1.49, c2=1.49):
        self.n_clusters = n_clusters
        if use_kmeans:
            k_means = KMeans(n_clusters=self.n_clusters)
            k_means.fit(data)
            self.centroids_pos = k_means.cluster_centers_
        else:
            self.centroids_pos = data[np.random.choice(list(range(len(data))), self.n_clusters)]

        #=========================================================================
        # Cada agrupamiento tiene un centroide que es el punto que lo representa
        # se asignan k datos aleatorios a k centroides
        #=========================================================================
        self.pb_val = np.inf
        
        #===============================================================
        # Mejor posicion personal para todos los centroides hasta aqui
        #===============================================================
        self.pb_pos = self.centroids_pos.copy()
        self.velocity = np.zeros_like(self.centroids_pos)
        
        #===============================================
        # Mejor agrupamiento de los datos hasta aqui
        #===============================================
        self.pb_clustering = None
        
        #===================================================================
        # Parametros del PSO (particle swarm optimization)
        #                    (optimizacion usando enjambres de particulas)
        #===================================================================
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_pb(self, data: np.ndarray):
        """
        Updates personal best score based on the fitness function mentioned in the paper (Equation(4))
        :return:
        """
        #==================================================================
        # Encuentra los datos (puntos) que pertenecen a cada agrupamiento
        # utilizando distancias a los centroides
        #==================================================================
        distances = self._get_distances(data=data)
        
        #==================================================================
        # La distancia minima entre los datos y un centroide indica que 
        # pertenece a ese agrupamiento
        #==================================================================
        clusters = np.argmin(distances, axis=0)  # shape: (len(data),)
        clusters_ids = np.unique(clusters)

        #=============================================================================
        # Si el algoritmo genera menos de n agrupamientos generar al azar la posicion
        # de un nuevo centroide para el id del agrupamiento que falta
        #=============================================================================
        while len(clusters_ids) != self.n_clusters:
            deleted_clusters = np.where(np.isin(np.arange(self.n_clusters), clusters_ids) == False)[0]
            self.centroids_pos[deleted_clusters] = data[np.random.choice(list(range(len(data))), len(deleted_clusters))]
            distances = self._get_distances(data=data)
            clusters = np.argmin(distances, axis=0)
            clusters_ids = np.unique(clusters)

        new_val = self._fitness_function(clusters=clusters, distances=distances)
        if new_val < self.pb_val:
            self.pb_val = new_val
            self.pb_pos = self.centroids_pos.copy()
            self.pb_clustering = clusters.copy()

    def update_velocity(self, gb_pos: np.ndarray):
        """
        Updates new velocity based on the current velocity, personal best position so far, and the swarm (global) best
        position so far.
        :param gb_pos: vector of best centroid positions among all particles so far
        :return:
        """
        self.velocity = self.w * self.velocity + \
                        self.c1 * np.random.random() * (self.pb_pos - self.centroids_pos) + \
                        self.c2 * np.random.random() * (gb_pos - self.centroids_pos)

    def move_centroids(self, gb_pos):
        self.update_velocity(gb_pos=gb_pos)
        new_pos = self.centroids_pos + self.velocity
        self.centroids_pos = new_pos.copy()

    def _get_distances(self, data: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia euclideana entre los datos y los centroides
        :param data:
        :return: distances: a numpy array of distances (len(centroids) x len(data))
        """
        distances = []
        for centroid in self.centroids_pos:
            # calcula distancia euclideana --> raiz de la suma de los cuadrados
            d = np.linalg.norm(data - centroid, axis=1)
            distances.append(d)
        distances = np.array(distances)
        return distances

    def _fitness_function(self, clusters: np.ndarray, distances: np.ndarray) -> float:
        """
        Calculates the fitness function ( Equation 4)
        i is the index of particle
        j is the index of clusters in the particle i
        p is the vector of the input data indices belonging the cluster[ij]
        z[p] is the vector of the input data belonging the cluster[ij]
        d is a vector of distances between z(p) and centroid j
        :param clusters:
        :param distances:
        :return: J:
        """
        J = 0.0
        for i in range(self.n_clusters):
            p = np.where(clusters == i)[0]
            if len(p):
                d = sum(distances[i][p])
                d /= len(p)
                J += d
        J /= self.n_clusters
        return J