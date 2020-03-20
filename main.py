
import numpy as np

from nn import Tnn

Num     = 30
x0      = np.linspace(0,1,Num)
y0      = np.linspace(0,1,Num)
xx,yy   = np.meshgrid(x0,y0)
#zz1      = np.sqrt(4**2.0 - (xx**2.0 + yy**2.0)) / 2.0
#zz2      = np.sqrt(2**2.0 - (xx**2.0 + yy**2.0)) / np.sqrt(2)

zz1     = xx*yy
zz2     = xx/2.0 + yy / 2.0

input = np.c_[xx.reshape((Num*Num,1)).copy(),yy.reshape((Num*Num,1)).copy()]
output= np.c_[zz1.reshape((Num*Num,1)).copy(), zz2.reshape((Num*Num,1)).copy()]

test_input = np.c_[np.random.random(10),np.random.random(10)]


if __name__ == '__main__':
    nt0 = Tnn(2,10,2)
    n_iteration = 100
    
    lr_cal = 1.0
    anneal = 0.999
       
    # start loop
    for i0 in range(n_iteration):
    
        err_cal = 0;
        for j0 in range(output.shape[0]):
            err_cal += nt0.train(input[j0,:], output[j0,:], lr_cal)
        
        #print 'Error: ', err_cal / output.shape[0], ' lr: ', lr_cal
        lr_cal *= anneal
        
        # validation
        test_prd = np.array([])
        
        for k0 in range(test_input.shape[0]):
            if(k0==0):
                test_prd = np.append(test_prd, nt0.predict(test_input[k0,:]))
            else:
                test_prd = np.vstack([test_prd, nt0.predict(test_input[k0,:])])
                
        # ground true
        test_gr  = np.c_[test_input[:,0].copy()*test_input[:,1].copy(), test_input[:,1].copy()/2.0 + test_input[:,0].copy()/2.0]
        
        err_val = 0;
        for k0 in range(test_prd.shape[0]):
            for m0 in range(output.shape[1]):
                err_val += 0.5*(test_prd[k0,m0] -test_gr[k0,m0])**2.0
                
        print 'Error: ', err_cal / output.shape[0], ' lr: ', lr_cal, 'Valid_Error: ', err_val / test_input.shape[0]
            
        
    
    
    