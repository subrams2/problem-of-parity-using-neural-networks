# Read the input and ground truth output files
x <- as.matrix(read.table(file = "input.csv", sep = ","))
y <- as.matrix(read.table(file = "output.csv", sep = ","))

# Necessary initializations
i <- l <- 1
j <- count <- m <- E <- delw1 <- delw2 <- 0
# Learning rate
eta <- 0.05
# Momentum term
alpha <- 0.99

# Variables to track eta 
eta.val = matrix(rep(0,30), nrow = 10, byrow = TRUE)
# Initialize error term
e = matrix(rep(1,16), nrow = 16, byrow = TRUE)
e.final = matrix(rep(0,160), nrow = 16, byrow = TRUE)

# Set a seed for reproducibility
set.seed(1)
# Initialize weight matrices
w1 = matrix(c(runif(20, -1, 1)), nrow = 5, byrow = TRUE)
set.seed(1)
w2 = matrix(c(runif(5, -1, 1)), nrow = 5)

while(eta <= 0.5) {
  repeat {
    # Iteration count
    j = j + 1
    
    # First hidden later
    z2 = x %*% w1
    # Activation function
    a2 = 1 / (1 + exp(-z2))
    # Append the initial bias term to the activation function
    b2 <- 1
    # Final function input for output layer
    a2 = cbind(a2, b2)
    
    # Output layer
    z3 <- a2 %*% w2
    # Final output
    yhat <- 1 / (1 + exp(-z3))
    # Error between ground truth and estimation
    e <- y - yhat
    # Mean Square Error (MSE)
    E[j] <- colSums((e)^2) / 16
    
    # Backpropagation at output layer
    z3.p <- yhat * (1 - yhat) 
    del3 <- - e * z3.p
    # Change in w2
    dedw2 <- eta * t(a2) %*% del3
    
    # Backpropagation at hidden layer
    z2.p <- a2 * (1 - a2) 
    del2 <- del3 %*% t(w2) * z2.p
    t.x <- t(x)
    # Change in w1
    dedw1 <- eta * t.x %*% del2
    dedw1 <- dedw1[,1:4]
    
    # Updating the initial weight matrices 
    w1 <- w1 - dedw1 + (alpha * delw1)
    w2 <- w2 - dedw2 + (alpha * delw2)
    delw1 <- - dedw1
    delw2 <- - dedw2
    
    # Setting a threshold for the minimum error in estimation
    for (k in 1:16) {
      if(e[k,] <= 0.05 && e[k,] >= -0.05) { 
        count <- count + 1
      }
    }
    
    # Breaking loop when all 16 inputs have reached the acceptable threshold error
    if (count == 16) {
      m <- m + 1
      e.final[,m] <- e
      count <- 0
      break
    }
    
    print("--------------------------------------")
    cat("Current epoch: ", j, "\n")
    print(count)
    count <- 0
  }
  # Create a dataframe to track iterations and MSE values for varying values of learning rate 
  eta.val[i,l] <- eta
  eta.val[i,l+1] <- j
  eta.val[i,l+2] <- E[j]
  i <- i + 1
  # Reset iteration counter
  j <- 0
  # Change to the next value of eta
  eta <- eta + 0.05
  # Reset terms for next iteration
  e <- matrix(rep(1,16), nrow = 16, byrow = TRUE)
  E <- 0
  set.seed(1)
  w1 <- matrix(c(runif(20, -1, 1)), nrow = 5, byrow = TRUE)
  set.seed(1)
  w2 <- matrix(c(runif(5, -1, 1)), nrow = 5)
}
cat("The algorithm has converged for all values of eta. Yay!", "\n", "The observations are listed below as eta:Epochs:MSE", "\n")
print(eta.val)
cat("The errors for all values of eta are: ", "\n")
print(e.final)
