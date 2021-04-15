import unittest
import sys
sys.path.append(".")
#from puissance4_objet import connect4
from ConnectFour1 import Board


class Testconnect4(unittest.TestCase):
    def test_verif_ligne(self):
        N = [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 1., 0., 0.]]
        seq = True
        board = Board(N)
        result = board.verif_ligne()
        #print('result =', result)
        self.assertEqual(result, seq) 
    
    def test_verif_colonne(self):
        Z = [[0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.]]
        seq = True
        board1 = Board(Z)
        result1 = board1.verif_colonne()
        #print('result1 = ',result1)
        self.assertEqual(result1,seq)

#test = Testconnect4()



if __name__ == '__main__':
    unittest.main()