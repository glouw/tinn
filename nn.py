import numpy as np

class Tnn:
    """my python verison of neural network"""
    
    nips=2
    nops=2
    nhis=10
    
    weights_hidden = np.array([])
    weights_output = np.array([])
    
    bias = np.array([])
    
    # value stored at each layer
    value_ops = np.array([])
    value_his = np.array([])
    
    def __init__(self, nips_0, nhis_0, nops_0):
        self.nips = nips_0
        self.nops = nops_0
        self.nhis = nhis_0
        
        self.weights_hidden = np.random.random(nips_0*nhis_0)
        self.weights_output = np.random.random(nhis_0*nops_0)
               
        self.bias = np.random.random(2)
        
        # value stored at each layer
        self.value_ops = np.zeros(nops_0)
        self.value_his = np.zeros(nhis_0)
    
    # activation function
    def actf(self, x_cal):
        return 1.0/(1.0+np.exp(-x_cal))
        
    # partial derivative of activate function
    def pdact(self, x_cal):
        return x_cal * (1.0 - x_cal)
        
    # partial derivative of error function
    def pderr(self, x_cal, y_cal):
        return x_cal - y_cal
        
    # error function
    def err(self, x_cal, y_cal):
        return 0.5*(x_cal-y_cal)*(x_cal-y_cal)
        
    # compute total error of the target to output
    def toerr(self, data_gt0, data_prd0):
        sum = np.sum(self.err(data_gt0, data_prd0))
        return sum
        
    def fprop(self, data_input):

        # for hidden layer    
        for i0 in range(self.nhis):
            sum = 0
            for j0 in range(self.nips):
                sum += data_input[j0]*self.weights_hidden[j0+i0*self.nips]
                
            self.value_his[i0] = self.actf(sum + self.bias[0])
            
        # for output layer
        for i0 in range(self.nops):
            sum = 0
            for j0 in range(self.nhis):
                sum += self.value_his[j0]*self.weights_output[j0+i0*self.nhis]
                
            self.value_ops[i0] = self.actf(sum + self.bias[1])
        
    def bprop(self, data_input, data_gt, lr):
        
        for i0 in range(self.nhis):
            sum = 0
            # update weights of output layer
            for j0 in range(self.nops):
                a_cal = self.pderr(self.value_ops[j0], data_gt[j0])
                b_cal = self.pdact(self.value_ops[j0])
                
                sum += a_cal * b_cal * self.weights_output[j0*self.nhis+i0]
            
                self.weights_output[j0*self.nhis+i0] -= lr * a_cal * b_cal * self.value_his[i0]
            
            # update weights in input layer            
            for j0 in range(self.nips):
                self.weights_hidden[i0*self.nips + j0] -= lr * sum * self.pdact(self.value_his[i0]) * data_input[j0]

    def predict(self, data_input_test):
        self.fprop(data_input_test)
        return self.value_ops
    
    def train(self, data_input0, data_gt0, lr0):
        self.fprop(data_input0)
        self.bprop(data_input0, data_gt0, lr0)
        return self.toerr(data_gt0, self.value_ops)
        
    def save_weights(self, name_s):
        np.savetxt(name_s, np.c_[self.weights_hidden, self.weights_output])
        
    
        
    