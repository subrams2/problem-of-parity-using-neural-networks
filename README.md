# Solving the Parity problem by building a neural network from scratch

1. The goal of this project is to implement a two-layer perceptron with the backpropagation algorithm to solve the parity problem. 

2. The desired output for the parity problem is 1 if an input pattern contains an odd number of 1’s and 0 otherwise. 

3. A simple neural network with 4 binary input elements, 4 hidden units for the ﬁrst layer, and one output unit for the second layer is used. 

4. The learning procedure is stopped when an absolute error (difference) of 0.05 is reached for every input pattern. 

5. All weights and biases are initialized to random numbers between -1 and 1.

6. A logistic sigmoid with a = 1 as the activation function is used for all units.

7. The value of η is varied from 0.05 to 0.5, incrementing by 0.05 each time.

8. The number of epochs and MSE for each choice of η is noted.

9. A momentum term α = 0.9 is included in another run of this algorithm and its eﬀect on the speed of training for each value of η is reported.
