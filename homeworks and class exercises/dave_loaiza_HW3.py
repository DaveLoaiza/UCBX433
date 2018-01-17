# -*- coding: utf-8 -*-
"""
Created on Tue May  2 23:57:57 2017

Part 1 – create your data:
1. Include a section line with your name

David Loaiza -- dave.loaiza@gmail.com

Python For Data Analysis And Scientific Computing
UC Berkeley Extension
"""

#2. Work only with these imports:
from numpy import matrix, array, random, min, max
import pylab as plb 

#3. Create a list A of 600 random numbers bound between (0:10)
A =  random.randint(0,11, size=600)     #upper bound not inclusive, range includes 0 and 10, 

#4. Create an array B with with 500 elements bound in the range [-3*pi:2*pi]
B = plb.linspace(-3*plb.pi,2*plb.pi, 500, endpoint=True)

#5. Using if, for or while, create a function that overwrites every element in A
# that falls outside of the interval [2:9), and overwrite
#that element with the average between the smallest and largest element in A
#6. Normalize each list element to be bound between [0:0.1]  
avg = (A.min() + A.max())/2

def update_array(X):
    for x in range(0, len(X)):
        if X[x] > 9 or X[x] < 2:
            X[x] = avg
    for x in range(0, len(X)):
        X[x] = ((plb.float64(X[x])-2)*(0.1))/(9-2)  #Did not fully understand step 6 request, used this formula: https://en.wikipedia.org/wiki/Feature_scaling#Rescaling
    return X

 
#7. Return the result from the functon to C
C = update_array(A)


#8. Cast C as an array
array(C)

#9. Add C to B (think of C as noise) and record the result in D … 
# (watch out: C is of different length. Truncate it)
D = B + C[:len(B)]

#Part 2 - plotting:
#10. Create a figure, give it a title and specify your own size and dpi
fig = plb.figure(figsize=(6,3), dpi = 240)
fig.canvas.set_window_title('Fun with plotting Sin, Cos, and Tan functions')

#11. Plot the sin of D, in the (2,1,1) location of the figure
plb.subplot(2,1,1)
plb.plot(D, plb.sin(D), color="red" )

#12. Overlay a plot of cos using D, with different color, thickness and type of line
plb.plot(D, plb.cos(D), color="blue", linewidth=2, linestyle="--")

#13. Create some space on top and bottom of the plot (on the y axis) and show the grid
plb.ylim(plb.sin(D).min() - 2, plb.sin(D).max() + 2)
plb.grid()


#14. Specify the following: Title, Y-axis label and legend to fit in the best way
plb.legend(loc='lower right')
plb.ylabel('amplitude', fontsize=10)
plb.title('sin and cos')

#15. Plot the tan of D, in location (2,1,2) with grid showing, X-axis label, Y-axis label 
# and legend on top right
plb.subplot(2,1,2)
plb.plot(D, plb.tan(D))
plb.grid()
plb.xlabel('period', fontsize=10)
plb.ylabel('amplitude', fontsize=10, )
plb.legend(loc='upper right')

#16. Organize your code: use each line from this HW as a comment line before coding each step
#17. Save these steps in a .py file and email it to me before next class. I will run it!
plb.show()