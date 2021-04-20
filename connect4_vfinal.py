# -*- coding: utf-8 -*-

import sys
import random 
import numpy as np
import pygame
import tensorflow as tf
from math import exp, log
#import random
from random import choice, uniform
from collections import deque
from keras import models
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from kaggle_environments import evaluate, make

class Board : 
    
    def __init__(self):
        self.M = np.zeros((6,7))
        self.GAGNER = 0
        self.matrice_pleine= 0
    
    def verif_ligne(self):
        temp = 0

        for i in range(6) :   # toutes les lignes
            for j in range(4) :   # glissement de la séquence  de 4
                temp = self.M[i][j]+self.M[i][j+1]+self.M[i][j+2]+self.M[i][j+3]
                if(temp==4):
                    self.GAGNER=1
               
                    break
                elif(temp==20) : 
                    self.GAGNER=2
           
                    break
                if(self.GAGNER != 0): break
            if(self.GAGNER != 0): break
    

    
    def verif_colonne(self):
        temp = 0
        
        for j in range(7) : #7 colonnes
            for i in range(3) : # glissement vertical de la séquence  
                temp = self.M[i][j]+self.M[i+1][j]+self.M[i+2][j]+self.M[i+3][j]
                if(temp==4):
                    self.GAGNER=1
                   
                    break
                elif(temp==20) : 
                    self.GAGNER=2
                   
                    break
                if(self.GAGNER != 0): break
            if(self.GAGNER != 0): break
       
            
            
    def verif_diagonale(self):
     
        for i in range(0,3):
            for j in range(0,4):
                temp = self.M[i][j]+self.M[i+1][j+1]+self.M[i+2][j+2]+self.M[i+3][j+3]
                if(temp==4):
                    self.GAGNER=1
                  
                    break
                elif(temp==20) : 
                    self.GAGNER=2
                    
                    break
                if(self.GAGNER != 0): break
            if(self.GAGNER != 0): break

        for i in range(5,3,-1):
            for j in range(0,4):
                temp=self.M[i][j]+self.M[i-1][j+1]+self.M[i-2][j+2]+self.M[i-3][j+3]
                if(temp==4):
                    self.GAGNER=1
                 
                    break
                elif(temp==20) : 
                    self.GAGNER=2
                   
                    break
                if(self.GAGNER != 0): break
            if(self.GAGNER != 0): break

class Memory:
    def __init__(self): 
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self): 
        self.observations = []
        self.actions = []
        self.rewards = []
        self.info = []
        
    def add_to_memory(self, new_observation, new_action, new_reward): 
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(float(new_reward))
        
        
        
class Deep_Q():
    def __init__(self): 
        self.player_1_model=None
        
    def check_if_done(self,observation):
        done = [False,'No Winner Yet']
        #horizontal check
        for i in range(6):
            for j in range(4):
                if observation[i][j] == observation[i][j+1] == observation[i][j+2] == observation[i][j+3] == 1:
                    done = [True,'Player 1 Wins Horizontal']
                if observation[i][j] == observation[i][j+1] == observation[i][j+2] == observation[i][j+3] == 2:
                    done = [True,'Player 2 Wins Horizontal']
        #vertical check
        for j in range(7):
            for i in range(3):
                if observation[i][j] == observation[i+1][j] == observation[i+2][j] == observation[i+3][j] == 1:
                    done = [True,'Player 1 Wins Vertical']
                if observation[i][j] == observation[i+1][j] == observation[i+2][j] == observation[i+3][j] == 2:
                    done = [True,'Player 2 Wins Vertical']
        #diagonal check top left to bottom right
        for row in range(3):
            for col in range(4):
                if observation[row][col] == observation[row + 1][col + 1] == observation[row + 2][col + 2] == observation[row + 3][col + 3] == 1:
                    done = [True,'Player 1 Wins Diagonal']
                if observation[row][col] == observation[row + 1][col + 1] == observation[row + 2][col + 2] == observation[row + 3][col + 3] == 2:
                    done = [True,'Player 2 Wins Diagonal']

        #diagonal check bottom left to top right
        for row in range(5, 2, -1):
            for col in range(3):
                if observation[row][col] == observation[row - 1][col + 1] == observation[row - 2][col + 2] == observation[row - 3][col + 3] == 1:
                    done = [True,'Player 1 Wins Diagonal']
                if observation[row][col] == observation[row - 1][col + 1] == observation[row - 2][col + 2] == observation[row - 3][col + 3] == 2:
                    done = [True,'Player 2 Wins Diagonal']
        return done
        
    def create_model(self):
        model = models.Sequential()

        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))

        model.add(Dense(7))

        return model


    def compute_loss(self,logits, actions, rewards): 
        neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
        loss = tf.reduce_mean(neg_logprob * rewards)
        return loss

    def train_step(self, model, optimizer, observations, actions, rewards):
        with tf.GradientTape() as tape:
          # Forward propagate through the agent network

            logits = model(observations)
            loss = self.compute_loss(logits, actions, rewards)
            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
    def get_action(self,model, observation, epsilon):
        #determine whether model action or random action based on epsilon
        act = np.random.choice(['model','random'], 1, p=[1-epsilon, epsilon])[0]
        observation = np.array(observation).reshape(1,6,7,1)
        logits = model.predict(observation)
        prob_weights = tf.nn.softmax(logits).numpy()

        if act == 'model':
            action = list(prob_weights[0]).index(max(prob_weights[0]))
        if act == 'random':
            action = np.random.choice(7)

        return action, prob_weights[0]
    
    def train_model(self,eps) :   
        #train player 1 against random agent
        LEARNING_RATE = 0.1
        tf.keras.backend.set_floatx('float64')
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

        env = make("connectx", debug=True)
        memory = Memory()
        epsilon = 1

        self.player_1_model = self.create_model()



        for i_episode in range(eps):
            print('i_episode = ', i_episode)
            trainer = env.train([None,'random'])  # contains step / reset methods

            observation = trainer.reset()['board']   #matrice zeros

            memory.clear()
            epsilon = epsilon * .99985
            overflow = False

            while True:
                action, _ = self.get_action(self.player_1_model,observation,epsilon)
                print('action = ', action, 'prob_weight = ', _)
                next_observation, dummy, overflow, info = trainer.step(action)
                observation = next_observation['board']
                #print(np.array(observation).reshape(6,7))
                observation = [float(i) for i in observation]
                done = self.check_if_done(np.array(observation).reshape(6,7))

                #-----Customize Rewards Here------
                if done[0] == False:
                    reward = 0
                if 'Player 2' in done[1]:
                    print('player 2 is the winner')
                    reward = -20
                if 'Player 1' in done[1]:
                    #win_count += 1
                    print('player 1 is the winner')
                    reward = 20
                if overflow == True and done[0] == False:
                    reward = -99
                    done[0] = True
                #-----Customize Rewards Here------

                memory.add_to_memory(np.array(observation).reshape(6,7,1), action, reward)
                if done[0]:
                    #train after each game

                    self.train_step(self.player_1_model, optimizer,
                             observations=np.array(memory.observations),
                             actions=np.array(memory.actions),
                             rewards = memory.rewards)

                    print('training is done', 'reward = ', reward)
                    break
                    
        self.player_1_model.save('Projets_Simplon/Connect4_f') 
        print('model_well_saved')
        
    def get_model(self): 
        return self.player_1_model
#     def save_model(self):
#         self.player_1_model.save('Projets_Simplon/Connect4') 
#         loaded = tf.keras.models.load_model('Projets_Simplon/Connect4')
    
    
# Create the base model from the pre-trained model MobileNet V2
# IMG_SHAPE = IMG_SIZE + (3,)
# base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
#                                                include_top=False,
#                                                weights='imagenet')

class Players(Deep_Q) : 
    
    def __init__(self):
        self.vs = None
        self.choice = None
        self.BLUE = (0,0,255)
        self.BLACK = (0,0,0)
        self.RED = (255,0,0)
        self.YELLOW = (255,255,0)
   
        
    def select_adversaire(self) : 
        oui = ["hxr","hxh","rxr"]
        if self.vs in oui:
                self.vs = print("please tape --help to get commands")
                #int(input("New Game \n 1 : vs computer \n 2 : vs friend \n "))
        if(self.vs=="hxr") : print("New Game with a computer will start, Good luck !")
        elif (self.vs=="hxh") : print("New Game with a friend will start, Good luck !")
        elif (self.vs=="rxr") : print("New Game between 2 robots will start, Good luck !")
    
    def game_interface(self):
        self.choice = input("Please choice your interface: \n T: Terminal  \n G: Graphical Interface  ")
        while (not(self.choice == "T" or self.choice == "G")):
            self.choice = input("Please choice your interface: \n T: Terminal \n G: Graphical Interface")
        print('new game will start !!! best of luck ')
         
    def detect(self):
        ROW_COUNT = 6
        COLUMN_COUNT = 7
        SQUARESIZE = 100
        RADIUS = int(SQUARESIZE/2 - 5)
        height = (ROW_COUNT+1) * SQUARESIZE
        width = COLUMN_COUNT * SQUARESIZE
        size = (width, height)
        screen = pygame.display.set_mode(size)
        consol_detect = False

        while consol_detect == False :
            for event in pygame.event.get(): 
                if event.type == pygame.MOUSEBUTTONDOWN:         
                    pygame.draw.rect(screen, self.BLACK, (0,0, width, SQUARESIZE))         
                    print(event.pos)         
                    posx = event.pos[0]    
                    posx1 = int(posx/100)
                    consol_detect = True
        return(posx1)     
    
    
    def IA_player(self,model,observation) : 
        print("tour du bot_IA")
        
        observation = np.array(observation).reshape(1,6,7,1)
        logits = model.predict(observation)
        prob_weights = tf.nn.softmax(logits).numpy()
        
        action, _ = self.get_action(model,observation,1)
        print('action = ', action, 'prob_weight = ', _)
        pos = action
      
        return pos
       
            
    def Random_player(self, columns) : 
        print('random_choice')
        pos = random.choice(columns)
        return pos
    
    
    def Human_player(self) : 
        if(self.choice == "T") :    
            pos = int(input('Which column do you want to play ? : '))
            while(not( pos>=0 and  pos<=7)) :  
                pos = int(input('Please choose a column number between 0 and 6 : ')) #pour jouer sur le terminal 
        elif(self.choice == "G") :
            pos= self.detect()
        return pos

class Game(Board, Players) :
    
    def __init__(self,vs):
        self.BLUE = (0,0,255)
        self.BLACK = (0,0,0)
        self.RED = (255,0,0)
        self.YELLOW = (255,255,0)
        self.choice = None
        self.vs = vs
        self.aqui = 1
        self.GAGNER = 0
        self.columns = [0,1,2,3,4,5,6]
        self.matrice_pleine= 0
        #joueur 1
        #joueur 2 

    def new_game(self) : # pour réinitialiser le jeu à chaque début de partie
        self.M = np.zeros((6,7))
        self.aqui = 1
        self.GAGNER = 0
        self.matrice_pleine= 0
        self.columns = [0,1,2,3,4,5,6]
    
    #à placer dans la classe Board
    def draw_board(self):
        ROW_COUNT = 6
        COLUMN_COUNT = 7
        SQUARESIZE = 100
        RADIUS = int(SQUARESIZE/2 - 5)
        height = (ROW_COUNT+1) * SQUARESIZE
        width = COLUMN_COUNT * SQUARESIZE
        size = (width, height)
        screen = pygame.display.set_mode(size)
        
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                pygame.draw.rect(screen, self.BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(screen, self.BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
                
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                if self.M[5-r][c] == 1:
                    pygame.draw.circle(screen, self.RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                elif self.M[5-r][c] == 5:
                    pygame.draw.circle(screen, self.YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
        pygame.display.update()
        pygame.init()
        

    
    
    
    def get_colonne(self):
        print('self.vs',self.vs)
        if(self.vs==1): 
            if(self.aqui==1) : 
                print("Player 1\'s turn ")
                pos = self.Human_player()
            else : 
                #pos = self.Random_player(self.columns)
#                 dqn=Deep_Q()
#                 model = dqn.get_model() #after save and load
                model = tf.keras.models.load_model('Projets_Simplon/Connect4_f') 
                pos = self.IA_player(model , self.M)
        elif(self.vs==2):
            if(self.aqui==1) : print("Player 1 turn")
            else : print("Player 2\'s turn")
            pos = self.Human_player()
        elif(self.vs==3):
            if(self.aqui==1) : 
                print("IA bot\'s turn")
                model = tf.keras.models.load_model('Projets_Simplon/Connect4_f') 
                pos = self.IA_player(model , self.M)
            else : 
                print("Random bot turn")
                pos = self.Random_player(self.columns)
            for i in range(10000000):
                pass
        return pos
            
        
    def placer_jeton(self,col):
        pos = col
        for i in range(5,-1, -1) :
            if(self.M[i][pos]==0) : 
                self.M[i][pos]=self.aqui
                if(self.choice== 'G') : self.draw_board()
                else : print(self.M)  
                break
            ## controle d'insertion
            if(i==0 and self.M[i][pos]!=0): 
                print("This column is full !!")
                self.columns=[]
                for j in range(7) : 
                    if  (self.M[0][j]==0) : self.columns.append(j)
                if(len(self.columns)>0) : 
                    print("Allowed columns : ", self.columns)
                    if(self.aqui==1): self.aqui= 5 #on doit rester sur le meme joueur
                    else: self.aqui=1

                else : 
                    print("Full grid ! End of the game :) ")
                    self.matrice_pleine=1
              
    def jouer(self) :
        self.new_game()
        self.select_adversaire()
        self.game_interface()
        if(self.choice == 'G') :
            self.draw_board()

        while(not(self.matrice_pleine) and self.GAGNER==0 ) :                       
            pos = self.get_colonne()
            
            ## insertion du jeton 
            self.placer_jeton(pos)
            ## un gagnant ?  
            self.verif_colonne()
            self.verif_ligne()
            self.verif_diagonale()

            if (self.GAGNER!=0):
                print('The winner is = ', self.GAGNER)
                for i in range(10000000):
                    pass
                break

            ##changement de joueur
            if(self.aqui==1): self.aqui=5
            else: self.aqui=1
                
        print('The winner is : ', self.GAGNER )

#dqn = Deep_Q()
#dqn.train_model(10)

#G = Game()
#G.jouer()


if __name__ == "__main__":
    
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>3}: {arg}")
    if len(sys.argv) == 2 and sys.argv[1]=="--help":
        print("To play against the robot, please type : python connect4_vfinal.py --mode hxr")
        print("To play against a friend, please type :  python connect4_vfinal.py --mode hxh")
    elif len(sys.argv) == 3:
        if sys.argv[1]== "--mode": 
            if sys.argv[2]== "hxr":
                G = Game(1)
                #G = Game(sys.argv[2]) => __init__(self, typeGame)
                G.jouer()
            elif sys.argv[2]== "hxh":
                G = Game(2)
                #G = Game(sys.argv[2]) => __init__(self, typeGame)
                G.jouer()
            elif sys.argv[2]== "rxr":
                G = Game(3)
                #G = Game(sys.argv[2]) => __init__(self, typeGame)
                G.jouer()
                
        else:
            print ("this argument is not allowed")
    else :
        print ("please tape --help as argument to read the doc")



