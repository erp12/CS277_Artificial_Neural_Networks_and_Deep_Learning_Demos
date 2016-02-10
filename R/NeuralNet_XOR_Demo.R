library(dplyr)
library(neuralnet)

# Create dataframe of all XOR patterns
input_a <- c(0, 1, 0, 1)
input_b <- c(0, 0, 1, 1)
XORoutput  <- c(0, 1, 1, 0)
XORdata <- data.frame(input_a, input_b, XORoutput)

# Get the names of the input and output values
n <- names(XORdata)

# Create, and train the neural network. Uses backpropagation.
# Arguments:
#   hidden = a vector of integers, each element gives the number of nodes in a hidden layer.
#   linear.output = If TRUE, regression. If FALSE, classification.
#   act.fct = Transfer function.
#   stepmax = Max number of epochs to train.
nn <- neuralnet(XORoutput~input_a+input_b, data=XORdata, hidden=c(3), linear.output=FALSE, act.fct="tanh", stepmax=1000)

# Plot the neural network topology, with weights and all!
plot(nn)

# Lets check each XOR pattern's output!
compute(nn,XORdata[1:2])
