#=================================================
# Cerebro del agente usando Pytorch
#=================================================
#Stephania Valdivia Diaz
#Fundamentos de IA
#=================================================

#==========================
# Modulos necesarios
#==========================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

#==========================
# Red neuronal lineal Q
#==========================
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

#========================================
# Aprendizaje (optimizacion)
#========================================
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    #=========================
    # Un entrenamiento
    #=========================
    def train_step(self, state, action, reward, next_state, done):
        
        #========================================
        # Convertir vectores a "tensores"
        #========================================
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        
        #===================================
        # Aprendizaje de corto plazo
        #===================================
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        #=================================================
        # 1: prediccion de valores Q con el estado actual
        #=================================================
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                #============================================
                # Ecuacion de Bellman (calidad de accion)
                # MODELO!
                #============================================
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                
            #=================================================
            # Guardar Q_new en matriz target (len(done) x 3) 
            #=================================================
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        #============================================================
        # 2: Q_new = R + gamma*max(next_predicted Q value) 
        # -> hacerlo si no ha terminado
        #============================================================
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        
        #===============================
        # Calculo del gradiente
        #===============================
        loss.backward()
        
        #=====================================
        # Optimizacion (avanzar al minimo)
        #=====================================
        self.optimizer.step()