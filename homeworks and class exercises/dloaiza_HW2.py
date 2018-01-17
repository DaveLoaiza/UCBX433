# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:05:55 2017

Homework Assignment #2:

1. Include a section with your name

@author: Dave Loaiza -- dave.loaiza@gmail.com

Python For Data Analysis And Scientific Computing
UC Berkeley Extensoin

16. Organize your code: use each line from this assignment as a comment line before each step
17. Save all steps as a script in a .py file
18. Email me your .py file and screenshots of your running code before next class. I will run it!
    
"""

import numpy as np
name = "Dave Loaiza"

#2. create matrix A with size (3,5) containing random numbers
A = np.random.random(15)
A = A.reshape(3,5)

#3. Find the size and length of matrix A
A.size
A.shape
len(A)

#4. Resize (crop) matrix A to size (3,4)
A = A[0:3,0:4]

#5. Find the transpose of matrix A and assign it to B
B = A.transpose()

#6. Find the minimum value in column 1 of matrix B
min(A[:,0])

#7. Find the minimum and maximum values for the en4re matrix A
A.min()
A.max()

#8. Create Vector X (an array) with 4 random numbers
X = np.random.random(4)

#9. Create a func4on and pass Vector X and matrix A in it
#10. In the new func4on mul4ply Vector X with matrix A and assign the result to D
#(note: you may get an error! … think why and fix it. Recall matrix manipula4on in class!)
def function(X,A):
    D = X*A

#11. Create a complex number Z with absolute and real parts != 0
    Z = np.csingle(73 + 142.14132j)

#12. Show its real and imaginary parts as well as it’s absolute value
    print(Z.real)
    print(Z.imag)
    print(np.absolute(Z))

#13. Muliply result D with the absolute value of Z and record it to C
    C = D*np.absolute(Z)

#14. Convert matrix B from a matrix to a string and overwrite B
B = str(B)

function(X,A)
#15. Display a text on the screen: ‘Name is done with HW2‘, but pass your ‘Name’ as a string variable
print(name + " is done with HW2")
