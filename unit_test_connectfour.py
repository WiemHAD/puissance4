import unittest
import sys
sys.path.append(".")
#from puissance4_objet import connect4
from connect4 import Board, Game


class Test_board(unittest.TestCase):
    def test_verif_ligne(self):
        N = [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 5., 5., 5.],
        [5., 5., 5., 1., 1., 1., 5.],
        [5., 1., 5., 5., 1., 1., 5.]]
        board = Board(N)
        result = board.verif_ligne()
        self.assertEqual(result, True) 
    
    def test_verif_colonne(self):
        Z = [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.]]

        board1 = Board(Z)
        result1 = board1.verif_colonne()
        self.assertEqual(result1,True)

    def test_verif_diag(self):
        Y = [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 1., 0., 0., 0.],
        [0., 1., 1., 5., 0., 0., 0.],
        [1., 1., 1., 5., 1., 1., 0.]]
        board2 = Board(Y)
        result2 = board2.verif_diagonale()
        self.assertEqual(result2, True)

    



'''test fonctionnel:
test: user case : cas gagnant
input: 1 matrice entree + 1 matrice sortie
2 joueurs
Boucle :
 - Etape 1: joueur 1
 - Etape 2: vérifier que le joueur 2 prend la main
 - Etape 3: Joueur 2 : joue
 - Etape 4: Vérifier que la case (colonne/ ligne) est disponible
 - Etape 5: vérifier que la bonne case a été jouée
 - Etape 6: redonner la main au joueur 2
 - Etape 7: reverifier la boucle précédente
 - Etape 8 : redonner la main au joueur 3
 - Etape 9 : reverifier la boucle précédente
 - Etape 10 : verifier les positions (diagonales / alignements) pas de force 4 '''

       
class Test_jeton(unittest.TestCase):
    def placer_jeton(self,M,col):
        pos = col
        matrice_pleine = 0
        for i in range(5,-1, -1) :            
            if(M[i][pos]==0) : 
                M[i][pos]=1             
            break
            ## controle d'insertion
            if(i==0 and M[i][pos]!=0): 
                print("La colonne est pleine !!")
                columns=[]
                for j in range(7) : 
                    if  (M[0][j]==0) : columns.append(j)
                if(len(columns)>0) : 
                    print("les colonnes possibles : ", columns)
                else : 
                    print("Matrice pleine ! fin du jeu :) ")
                    matrice_pleine=1
        return(M)

    def test_placer_jeton(self):
        X = [[1., 5., 5., 1., 1., 1., 5.],
        [5., 5., 1., 5., 1., 1., 1.],
        [5., 1., 1., 5., 5., 5., 1.],
        [1., 1., 5., 1., 1., 5., 5.],
        [5., 1., 5., 5., 1., 1., 5.],
        [1., 5., 5., 1., 5., 5., 1.]]

        M = [[1., 5., 5., 1., 1., 1., 5.],
        [5., 5., 1., 5., 1., 1., 1.],
        [5., 1., 1., 5., 5., 5., 1.],
        [1., 1., 5., 1., 1., 5., 5.],
        [5., 1., 5., 5., 1., 1., 5.],
        [1., 5., 5., 1., 5., 5., 1.]]

        col = 3
        result3 = self.placer_jeton(X,col)
        #print(result3)
        self.assertEqual(result3, M)

    def test_placer_jeton1(self):
        X1 = [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]]

        M1 = [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.]]
        
        col = 3
        result4 = self.placer_jeton(X1,col)
        self.assertEqual(result4, M1)

if __name__ == '__main__':
    unittest.main()