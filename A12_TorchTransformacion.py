#=================================================
# Transformaciones de Tensores
#=================================================
#Stephania Valdivia Diaz
#Fundamentos de IA
#=================================================
'''
Transformaciones pueden ser aplicadas a imagenes PIL, tensores, ndarrays
o datos comunes durante la creacion de la base de datos.

Lista completa de transformaciones ya programadas:
    https://pytorch.org/docs/stable/torchvision/transforms.html

En imagenes
============
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

En Tensores
============
LinearTransformation, Normalize, RandomErasing

Conversiones
=============
ToPILImage: convertir de tensor a numpy ndarray
ToTensor: de numpy.ndarray a PILImage

Generico
=========
Usar Lambda

Comunes
========
Escribir tu propio objeto (clase)

Componer (compose) multiples transformaciones
==============================================
composed = transforms.Compose([Rescale(256), RandomCrop(224)])
'''

#==========================
# Modulos necesarios
#==========================
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

#======================================
# Clase WineDataset hija de Dataset
#======================================
class WineDataset(Dataset):
    
    #===============
    # Constructor
    #===============
    def __init__(self, transform=None):
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape [0]
        
        # note que no convertimos en tensor aqui
        self.x_data = xy[:, 1:]   
        self.y_data = xy[:, [0]]
        
        self.transform = transform
        
    #=============================
    # Metodo para obtener datos
    #=============================
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    #================================
    # Tamaño del conjunto de datos
    #================================
    def __len__(self):
        return self.n_samples
    
#============================
# Transformaciones comunes
#============================

#============================
# De numpy a tensor pytorch
#============================
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
#===================================================
# Escalar datos (multiplicarlos por una constante)
#===================================================
class MulTransform:
    
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets
    
#======================
# Programa principal
#======================
if __name__ == "__main__":
    
    print('Sin transformacion')
    dataset = WineDataset()
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)
    
    print('\nTransformando en tensor')
    dataset = WineDataset(transform=ToTensor())
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)
    
    print('\nCon transformacion a tensor y multiplicacion')
    composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
    dataset = WineDataset(transform=composed)
    first_data = dataset[0]
    features, labels = first_data
    print(type(features), type(labels))
    print(features, labels)
    