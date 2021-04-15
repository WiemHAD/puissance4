import random 
import numpy as np
import pygame


M1 = np.zeros((6,7))
class Board : 
    
    def __init__(self):
        self.M = np.zeros((6,7))
        self.GAGNER = 0
        self.matrice_pleine= 0
        
    def verif_ligne(self):
        temp = 0
        for i in range(6) :   # toutes les lignes
            for j in range(3) :   # glissement de la séquence  de 4
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

class Players() : 
    def __init__(self):
        self.vs = None
        self.choice = None
        self.BLUE = (0,0,255)
        self.BLACK = (0,0,0)
        self.RED = (255,0,0)
        self.YELLOW = (255,255,0)
    
    def select_adversaire(self) :
        while(not( self.vs==1 or  self.vs==2)) :  
                self.vs = int(input("New Game \n 1 : vs computer \n 2 : vs friend \n "))
        if(self.vs==1) : print("New Game with a computer will start, Good luck !")
        elif (self.vs==2) : print("New Game with a freind will start, Good luck !")
    
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
    
    
    def IA_player(self, columns) : 
        print("tour du bot")
        pos = random.choice(columns)
        return pos
    

    def Human_player(self) : 
        if(self.choice == "T") :    
            pos = int(input('A quelle colonne voulez-vous jouer: '))
            while(not( pos>=0 and  pos<=7)) :  
                pos = int(input('choisir une colonne entre 0 et 7: ')) #pour jouer sur le terminal 
        elif(self.choice == "G") :
            pos= self.detect()
    
        return pos

class Game(Board, Players) :
    
    def __init__(self):
        self.aqui = 1
        self.choice = None
        self.BLUE = (0,0,255)
        self.BLACK = (0,0,0)
        self.RED = (255,0,0)
        self.YELLOW = (255,255,0)
        #joueur 1
        #joueur 2 

    def new_game(self) : 
        self.M = np.zeros((6,7))
        self.aqui = 1
        self.GAGNER = 0
        self.matrice_pleine= 0
        self.vs = None
        self.columns = [0,1,2,3,4,5,6]
        

        
    def get_colonne(self):
        if(self.vs==1): 
            if(self.aqui==1) : 
                print("tour du  joueur 1 ")
                pos = self.Human_player()
            else : 
                pos = self.IA_player(self.columns)
                
        elif(self.vs==2):
            if(self.aqui==1) : print("tour du joueur 1")
            else : print("tour du jour 2")
          
            pos = self.Human_player()
        
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
                print("La colonne est pleine !!")
                self.columns=[]
                for j in range(7) : 
                    if  (self.M[0][j]==0) : self.columns.append(j)
                if(len(self.columns)>0) : 
                    print("les colonnes possibles : ", self.columns)
                    if(self.aqui==1): self.aqui= 5 #on doit rester sur le meme joueur
                    else: self.aqui=1

                else : 
                    print("Matrice pleine ! fin du jeu :) ")
                    self.matrice_pleine=1
              
    def jouer(self) :
        self.new_game()
        self.select_adversaire()
        self.game_interface()
        if(self.choice == 'G') :
            self.draw_board()
        #self.detect()
    

        
        while(not(self.matrice_pleine)):                       
            pos = self.get_colonne()
            #pos = self.detect()
            ## insertion du jeton 
            self.placer_jeton(pos)
            ## un gagnant ?  
            self.verif_colonne()
            self.verif_ligne()
            self.verif_diagonale()

            if (self.GAGNER!=0): 
                print('le gagnant est = ', self.GAGNER)
                break

            ##changement de joueur
            if(self.aqui==1): self.aqui=5
            else: self.aqui=1

        print('le gagnant est le joueur : ', self.GAGNER )

        

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
    
  

G = Game()
G.jouer()

