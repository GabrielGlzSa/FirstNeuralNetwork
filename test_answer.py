import numpy as np
import random
import unittest


from notebooks.exercise import relu_function, relu_function_layer
from notebooks.exercise import calculate_neuron_logit, calculate_layer_logit
from notebooks.exercise import softmax_layer
from notebooks.exercise import weight_initialization, neural_network


class TestExercises(unittest.TestCase):
    
    def test_relu_function(self):
        self.assertEqual(relu_function(0), 0)
        self.assertEqual(relu_function(-3), 0)
        self.assertEqual(relu_function(2), 2)
        neg_value = random.randint(-100,-1)
        self.assertEqual(relu_function(neg_value), 0)
        pos_value = random.randint(1,100)
        self.assertEqual(relu_function(pos_value), max(0, pos_value))


    def test_relu_layer(self):
        x = np.random.uniform(-10, 10, 100)
        self.assertEqual(relu_function_layer(x).tolist(), np.maximum(x, 0).tolist())

    def test_calculate_neuron_logit(self):
        x = np.arange(9).reshape(3,3)
        w = np.array([0.1, 0.2, 0.3])
        b = -0.5
        sum_wx = np.dot(w,x)
        logits = sum_wx + b
        self.assertEqual(calculate_neuron_logit(x, w, b), logits)


    def test_calculate_layer_logit(self):
        X = np.arange(9).reshape(3,3)
        W = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.0, 0.9]])
        b = np.array([-0.5, 0.7, 0.1])
        o = X @ W + b
        self.assertEqual(calculate_layer_logit(X, W, b).tolist(), o.tolist())

    def test_softmax_layer(self):
        X = np.arange(9).reshape(3,3)
        W = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.0, 0.9]])
        b = np.array([-0.5, 0.7, 0.1])
        logits = X @ W + b
        o = np.exp(logits) / np.sum(np.exp(logits), axis=1).reshape(-1,1)
        self.assertEqual(softmax_layer(X, W, b).tolist(), o.tolist())

    def test_weight_initialization(self):
        w1, w2, w3, b1, b2, b3 = weight_initialization(8, 32, 32, 4)

        self.assertEqual(w1.shape, (8, 32))
        self.assertEqual(w2.shape, (32, 32))
        self.assertEqual(w3.shape, (32, 4))
        self.assertEqual(b1.shape, (32,))
        self.assertEqual(b2.shape, (32,))
        self.assertEqual(b3.shape, (4,))

    def test_neural_network(self):
        w1, w2, w3, b1, b2, b3 = weight_initialization(8, 32, 32, 4)
        x = np.random.uniform(-10, 10, (10, 8))
        o1, o2, probabilities = neural_network(x, w1, b1, w2, b2, w3, b3)

        # Test correct output shape of first layer.
        self.assertEqual(o1.shape, (10, 32))
        
        t1 = np.maximum(x @ w1 + b1, 0)
        self.assertEqual(o1.tolist(), t1.tolist())

        # Test correct output shape of second layer.
        self.assertEqual(o2.shape, (10, 32))
        t2 = np.maximum(o1 @ w2 + b2, 0)
        self.assertEqual(o2.tolist(), t2.tolist())

        # Test correct output shape of probabilities.
        self.assertEqual(probabilities.shape, (10, 4))

        logits = o2 @ w3 + b3
        prob = np.exp(logits) / np.sum(np.exp(logits), axis=1).reshape(-1,1)
        
        self.assertEqual(probabilities, prob)


        

if __name__ == '__main__':
    unittest.main()
