x <- as.matrix(read.table(file = "input.csv", sep = ","))
y <- as.matrix(read.table(file = "output.csv", sep = ","))

i <- k <- l <- 1
j <- count <- E <- m <- 0
eta = 0.05
eta.val = matrix(rep(0,30), nrow = 10, byrow = TRUE)
e = matrix(rep(1,16), nrow = 16, byrow = TRUE)
e.final = matrix(rep(0,160), nrow = 16, byrow = TRUE)

set.seed(1)
w1 = matrix(c(runif(20, -1, 1)), nrow = 5, byrow = TRUE)
set.seed(1)
w2 = matrix(c(runif(5, -1, 1)), nrow = 5)

while(eta <= 0.5) {
  repeat {
    j = j + 1
    
    z2 = x %*% w1
    a2 = 1 / (1 + exp(-z2))
    b2 <- 1
    a2 = cbind(a2, b2)
    
    z3 = a2 %*% w2
    yhat = 1 / (1 + exp(-z3))
    e = y - yhat
    E[j] = colSums((e)^2) / 16
    
    z3.p = yhat * (1 - yhat) 
    del3 = - e * z3.p
    dedw2 = eta * t(a2) %*% del3
    
    z2.p = a2 * (1 - a2) 
    del2 = del3 %*% t(w2) * z2.p
    t.x = t(x)
    dedw1 = eta * t.x %*% del2
    dedw1 = dedw1[,1:4]
    
    w1 = w1 - dedw1
    w2 = w2 - dedw2
    
    for (k in 1:16) {
      if(e[k,] <= 0.05 && e[k,] >= -0.05) { 
        count <- count + 1
      }
    }
    
    if (count == 16) {
      m = m + 1
      e.final[,m] = e
      break
    }
    
    print("--------------------------------------")
    cat("Current epoch: ", j, "\n")
    print(count)
    count = 0
  }
  eta.val[i,l] = eta
  eta.val[i,l+1] = j
  eta.val[i,l+2] = E[j]
  i = i + 1
  j = 0
  count = 0
  eta = eta + 0.05
  e = matrix(rep(1,16), nrow = 16, byrow = TRUE)
  E = 0
  set.seed(1)
  w1 = matrix(c(runif(20, -1, 1)), nrow = 5, byrow = TRUE)
  set.seed(1)
  w2 = matrix(c(runif(5, -1, 1)), nrow = 5)
}
cat("The algorithm has converged for all values of eta.", "\n", "The observations are listed below as eta:Epochs:MSE", "\n")
print(eta.val)
cat("The errors for all values of eta are: ", "\n")
print(e.final)

