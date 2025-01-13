import numpy as np
import pandas as pd
import random


#########function##########
def f(x,y):
    return  5*x**2 + 40 * x +  y**2 - 12 * y + 127



##########partial derivative of x######
def dfx(x):
    return 10 * x + 40

###########partial derivative of y##################
def dfy(y):
    return 2 * y - 12

#########list of learning rates##############
learn = [0.1, 0.01, 0.001]

for lr in learn:
    for j in range(10):     # number of trials
        x_1 = random.uniform(-10,10)
        y_1 = random.uniform(-10,10)
    
        for i in range(500):          # number of iterations
            x_1 = x_1 - lr * dfx(x_1) 
            y_1= y_1 - lr * dfy(y_1)
        
            func = f(x_1,y_1)         ######replacing x nad y with x_1 an y_1
        
        print('trial', j + 1 ,':', ([x_1,y_1]))
        print('function of trial', j + 1 ,':', func)
   
            
 

    