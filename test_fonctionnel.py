import sys
sys.path.append(".")
from connect4 import Board

A =[[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 5., 5., 1., 0., 0.]]

B = [[0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 0., 0., 0., 0., 0.,],
    [0., 0., 1., 0., 0., 0., 0.,],
    [0., 0., 5., 0., 5., 0., 0.,],
    [5., 0., 5., 1., 1., 1., 1.,]]

C = [[0., 0., 0., 5., 0., 0., 0.,],
    [0., 0., 1., 5., 0., 0., 0.,],
    [0., 5., 1., 5., 1., 5., 0.,],
    [0., 1., 1., 5., 1., 1., 0.,],
    [0., 1., 5., 1., 5., 5., 0.,],
    [5., 5., 5., 1., 5., 1., 1.,]]

l_matrice = [A, B, C]


def verif_test_fonctionnel(A):
        r = False
        board = Board(A)
        test = False
        
        if board.verif_colonne() == True: 
            test = True 
        elif board.verif_ligne() == True:
            test = True
        elif board.verif_diagonale() == True:
            test = True 
        else: 
            print("la partie n'est pas gagante")
        
        if test == r: 
            print("test fonctionnel OK ")

for i in l_matrice:
    test = verif_test_fonctionnel(i)
    test