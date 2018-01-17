"""
This file includes all the Python programming examples given by
professor Alex Iliev during the Python for Numerical Analysis and 
Scientific Computing course given at UC Berkeley Extension, Spring 2017

"""


#check packages
import pip
pip

pip list

pip.get_installed_distributions()

help()   #help command

whos #display variables

a = [3,5,8]

type(a)  #out: list

history -n #display list of old commands

save session 2 6 #writes second, sixth command in history list to a file 'session.py'

logstart     #start logging a session anew

logoff     # temporarily stop logging

logon     # restart logging

logstate     #print the status of the logging 

logstop     # fully stop logging and close log file

import sys

sys.modules.keys() # last two list loaded modules

print()      # print command

# simple scipi example:

from scipy.fftpack import fft, ifft
from numpy import array
a = array([2.3, 3.1, 4.2, -1.8, 1.6, 5.9])
a
b = fft(a)
b_inverse = ifft(b)
a_sum = a.sum()

#simple example 2

from scipy.opack import o
import numpy as np
import matplotlib.pyplot as plt
F = 600 # Sample frequency:
T = 1.0 / F # Period T:
a = np.linspace(0.0, F*T, F)
b = np.sin(100.0 * 4.0*np.pi*a) + 0.6*np.sin(70.0 * 3.0*np.pi*a)
c = np.cos(50.0 * 2.0*np.pi*a) + 0.8*np.sin(60.0 * 2.0*np.pi*a)
bf = o(b); cf = o(c)
af = np.linspace(0.0, 1.0/(3.0*T), F/2)
plt.plot(af, 3.0/F * np.abs(bf[0:F/2]),'r--')
plt.plot(af, 2.5/F * np.abs(cf[0:F/2]),'b-')
plt.grid(); plt.pause(1) # plt.show()

# simple example 3
ll = plt.plot(x,y)
xl = plt.xlabel('horizontal axis')
yl = plt.ylabel('ver4cal axis')
[l = plt.4tle('sine func4on')
ax = plt.axis([-2, 12, -1.5, 1.5])
grd = plt.grid(True)
txt = plt.text(0,1.3,'here is some text')
ann = plt.annotate('a point on curve',xy=(4.7,-1),xytext=(3,-1.3), arrowprops=dict(arrowstyle='->'))
plt.pause(1) # plt.show()

# simple example 4
x = arange(0.,10,0.3)
a = sin(x); b = cos(x);
c = exp(x/10); d = exp(-x/10)
plt.plot(x,a,'b-',label='sine')
plt.plot(x,b,'r--',label='cosine')
plt.plot(x,c,'c-.',label='exp(+x)')
plt.plot(x,d,'gx-', linewidth = 1.5,label='exp(-x)')
plt.legend(loc='upper lej')
plt.grid()
plt.xlabel('xaxis')
plt.ylabel('yaxis')
plt.pause(1) # plt.show()

# lecture 2
a = 128
type(a)  # = int
id(a) # object id

#six built constants:

q = None
w = NotImplemented
e = True
e = False

#tuples, lists, dictionaries

tp1 = (1,2,3,4,5)  #create tuple

min(tp1)   # 1
max(tp1)	#5

ls1 = [1,2,3,4,5]  # create list

ls1[0:3]       	# [1,2,3]

del ls1[3]      # ls1 now = [1,2,3,5]

list(tp1)  		# converts to a list
ls1.index(5) # returns 3
ls1.count(5) # returns 1
ls1.append(4) # appends one object (int 4) at the end of a list
ls1.pop(-3) # removes 3 from list and returns 3
ls1.extend(seq) # appends a sequence to a list
ls1.sort([funct]) # sorts all objects in a given list with given ‘funct’
ls1.insert(ind,obj) # inserts a par4cular object to a list with an offset index
ls1.remove(obj) # removes a par4cular object from a list
ls1.reverse() # reverse the order of objects in a list

dictionary1 = {'name': 'dave', 'age': 32}

dictionary2 = {}   #creates empty dictionary1

print(dictionary1.keys())   	#name, age
print(dictionary1.values()) 	#dave, 32

#memory allocation usage

getsizeof(ls1)

getsizeof(float)

getsizeof(True)

getsizeof(NotImplemented)

#checking validity

34 == 34      #True

34 == 35      #False

1 in ls1      #True

'Dave' in ls1     #False

34 is 34     # True

34. is 34    #False

#simple for lops
for k in ('Sarah', 'cars', 'Python'):
	print('John likes %s' % k)

for j in range(3):
	print(j)

def new_range(start, end, step):
	while start <= end:
		yield start # yeld is a generator preserving funct. local value
		start += step

for x in new_range(2, 5, 0.2): print(x)

x = [41, 12, 34, 52]
for k in x:
	if k==34:
		continue
	print(k)

#while loop and break
a = 6 + 4.5j
b = 1
while b<a.real:
	a = a**0.5+0.3
	print(a)
	print(b)
	b=b+1
	if a.imag < 0.5:
		print('The imaginary part fell below 0.5. Will exit now!')
		break

# functions work with local and global variables

a = 12
def dave_fun_test(b):		
	c = 41
	return a + b + c		#call dave_fun_test(2) returns 55

#optional parameters in functions
def fun_optional(d=12):
	return d + 34
# d = 12 unless a different value is passed in functional call

''' classes
 - methods are fuctions that are members of a class
 - they are functions attached to objects
 - a class can be called with different methods(functions) that it consists of
 '''
 
 #simple class
 class simple_class:
	"""this class shows basic functinality"""
	a = 12
	def f():
		return 'hello world'
		
simple_class.f() 	#'hello world'

simple_class.a = 9

class_1.a     #returns 9

class_1.__doc__      #'This class shows basic functionality'

# Class inheritance

# base class
class class_test:
	def method_one():
		print('This is method 1')
	def method_two():
		print('This is method 2')

# subclass
class class_test_two(class_test):
	def method_one():
		print('This is new method 1')	#method overriding from super class
	def method_three():
		print('this is method 3')
		
# class polymorphism

class Animal:
	def __init__(self, name):		#Constructor of the class
		self.name = name
	def talk(self):
		raise NotImplementedError('Subclass must implement abstract method')
	
class Cat(Animal):
	def talk(self):
		return 'Meow!'

class Dog(Animal):
	def talk(self):
		return('Woof! Woof!')
		
animals = [Cat('Tiger'), Cat('Kitty'), Dog('Maxie')]

def animal_sounds():
	for animal in animals:
		print(animal.name + ': ' + animal.talk())

# * import --- imports everything except private classes and functions
from sys import *		
		
public_var = 12
_private_var = 34

def public_fun():
	print('this function is public')
def _private_fun():	
	print('... now this function is set to be private')

class PublicClass():
	print('This is a public Class')
class _PrivateClass():
	print('... now this class is private')

# try a star import and a whos to see only public stuff imported
 
#I/O interactin with files
#open file modes: r - read-only | w - write only | r+ - read-write | a - append
file = open(files/lecture3/test.txt, 'r') #opens file in read mode
sentences = file.readlines()
print(sentences) 
print(len(sentences))
file.close()

file - open('files/lecture3/test.txt', 'w')  # opens file in write mode
file.write('We will overwrite the previous text \n and go to a new line as well')
file.close()

file = open('files/lecture3/test.txt', 'r')
sentences = file.readlines()
print(sentences) 
print(len(sentences))
file.close()

"""
Standard Library
• Some of the top standard library modules in Python are:
• Os – provides a selected list of opera4ng system level func4onality
• Sys – provides access to some variables used by the interpreter
• Io – deals with I/O func4onality for the three main types of I/O: text, binary and raw
• Math – it gives access to mathema4cal func4ons excluding complex numbers (->cmath)
• Wave – part of Python core installa4on, provides interface to the WAV format
• Audioop – consist of useful tools for operating on digital sound sampled data
• Html – provides an u4lity to work with the html language
• Time – provides func4ons related to time
• Calendar – provides various calendar capability
• Daytime – extended way of manipulating date and time
"""

#catching exceptions:

while True:
	try:
		a = int(input('Please Enter a number: '))
		print('You entered the number ', a)
		print('I will now exit. Good Bye!')
		break
	except ValueError:
		print('You entered an invalid number. Please try again')

#numpy arrays
a = np.array([[12, 34, 41], [54, 62, 18], [72, 84, 96]])

a.size  		#get size of array
.
a.shape  		#get shape of array, 3 x 3

type(a)  		#find type of object a**0

a.dtype  		# find data type of object a contents

a[2,2]   		#indexing of particular element (96) -- starts from position 0

b=a[0,:]  		# b = [12, 34, 41]

b[2] = 88  	    #reassigns a value, change affects the original array

a  		#check values of both variables
b

c= np.zeros(shape=4,5))   #creates a 4x5 array of all zeros

d = np.empty(shape=(2,2))  #creates empty array

#endpoint option:

plb.linspace(1,5,10, endpoint =False)	#array([1,1.4,1.8,2.2,2.6,3,3.4,3.8,4.2,4.6])

plb.linspace(1,5,10, endpoint=True)	#array([1,1.444, 1.888,2.333,2.777,3.222,3.6666,4.1111,4.5555, 5])

"""
List of NumPy Array Attributes:
T 	Same as self.transpose(), except that self is returned if self.ndim < 2.
data	 Python buffer object poin4ng to the start of the array’s data.
dtype 	Data-type of the array elements.
flags 	Informa4on about the memory layout of the array. (more info slide 15, Lecture 4)
flatten 	A 1-D iterator over the array.
imag 	The imaginary part of the array.
real 	The real part of the array.
size 	Number of elements in the array.
itemsize	 Length of one array element in bytes.
nbytes 	Total bytes consumed by the elements of the array.
ndim 	Number of array dimensions.
shape 	Tuple of array dimensions.
strides 	Tuple of bytes to step in each dimension when traversing an array.
ctypes 	An object to simplify the interac4on of the array with the ctypes module.
base 	Base object if memory is from some other object.
"""

#Exercise:
from numpy import array as ar
from numpy import matrix as mx
from numpy import int16, random

class _file_operations():
	def write_my_file(A):
		file = open('my_array', 'w')
		file.writelines('%s' %str(A))
		file.close()
		
	def read_my_file():
		file = open('my_array', 'r')
		B = file.read()
		file.close()
		return B

B = random.randint(4,45,6)
B.tofile('my_array', sep=',', format='%s')
B = _file_operations.read_my_file()
B = ar(B.split(','), dtype=int16)
C = mx([[3],[2],[5]])
D = C*B
_file_operations.write_my_file(D)
print(_file_operations.read_my_file())

"""
HW assignment 2 Official solution below
1. Include a sec4on with your name
2. Create matrix A with size (3,5) containing random numbers
3. Find the size and length of matrix A
4. Resize (crop) matrix A to size (3,4)
5. Find the transpose of matrix A and assign it to B
6. Find the minimum value in column 1 of matrix B
7. Find the minimum and maximum values for the en4re matrix A
8. Create Vector X (an array) with 4 random numbers
9. Create a func4on and pass Vector X and matrix A in it
10. In the new func4on mul4ply Vector X with matrix A and assign the result to D
(note: you may get an error! … think why and fix it. Recall matrix manipula4on in class!)
11. Create a complex number Z with absolute and real parts != 0
12. Show its real and imaginary parts as well as it’s absolute value
13. Mul4ply result D with the absolute value of Z and record it to C
14. Convert matrix B from a matrix to a string and overwrite B
15. Display a text on the screen: ‘Name is done with HW2‘, but pass your ‘Name’ as a string variable
16. Organize your code: use each line from this assignment as a comment line before each step
17. Save all steps as a script in a .py file
18. Email me your .py file and screenshots of your running code before next class. I will run it!
"""

from numpy import matrix, array, random, min, max	#1

A = random.random(15)	#2
A = A.reshape(3,5)
A = matrix(A)

A.size	#3
len(A)

A = A[0:3,0:4]	#4

B = A.T	#5

B[:,1].min()	#6

A.min()	#7
A.max()

X = array([random.random(4)])	#8

def function_hw2(a,b):	#9
	return a*b.T 

D = function_hw2(X,A)	#10

Z = 6+5J	#11

Z.real		#12
Z.imag
abs(Z)

C = D*abs(Z)		#13

B = str(B)		#14

print('%s is done with HW2' %'Dave')

#multiple string variable substitution:
A  = 'Dave'
B = 'Loves'
C = 'pizza'

print('If one thing is true, it is that %s %s %s.' %(A, B, C))

# prints 'If one thing is true, it is that Dave loves Pizza'

#matrix multiplication of an ndarray:
A = array([2,3,4],[4,5,6])
dot(A, A.T)

# matrix vs array vs ndarray in numpy
A = array([2,3,4],[4,5,6])
B = matrix([[2,3,4],[4,5,6]])
C = ndarray([2,3], dtype=int16)
A
B
C
C[0,:] = A[0,:]
C[1,:] = B[1,:]
C
type(A)
type(C)
A*A
B*B.T
C*C
dot(A,A.T)
dot(C,C.T)

# MATPLOTLIB and basic plotting

import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(-.np.pi*2, np.pi*2, 512, endpoint=True)
b_sin, c_cos = np.sin(a), np.cos(a)

plt.plot(a, b_sin)
plt.plot(a, c_cos)
plt.show()

plb.figure(figsize=(10,6), dpi = 120)	#create figure 10x6 inches with dpi=120
plb.subplot(3,2,6) #divides figure window into 3x2, places subplot in spot 6
import pylab as plb
d = plb.linspace(-plb.pi*3, plb.pi*3, 128, endpoint=True)
d_sin = plb.sin(d)
d_cos = plb.cos(d)

#plot 'sin' using a green dash-doted line of width 1.5px in area 2,1,1
plb.subplot(2,1,1)
plb.plot(d, d_sin, color="green", linewidth=1.5, linestyle='-.', label="sin")

#plot 'cos' using a blue dashed line of width 1.5 px in area 3,2,6
plb.subplot(3,2,6)
plb.plot(d, d_cos, color='blue', linewidth=1.5, linestyle='--', label="cos")

# set x limits
plb.xlim(-8.0, 8.0)
# plot x axis ticks
plb.xticks(plb.linspace(-8,8,6, endpoint=True)

# now we set the 'y' limits and ticks:
plb.ylim(-1.2, 1.4)
plb.yticks(plb.linspace(-1.2, 1.4, 4, endpoint=True)

#and save the figure
plb.savefig("lecture_5.png", dpi=64)

#setting explicit tick labels
plb.yticks([-1, 0, 1], ['$-1$', '$0$', '$+1'])
plb.xticks([-plb.pi*2, -3*plb.pi/2, -plb.pi, -plb.pi/2, 0, plb.pi/2, plb.pi, 3*plb.pi/2, plb.pi*2],['$-2\pi$', '$-3\pi/2$', '$-\pi$', '$\pi/2$', '$0$', '$\pi/2$', '$+\pi$', '$3\pi/2$', '$+2\pi$'])

#add legend in upper right corner
plb.legend(loc='upper right')

ax1 = plb.gca()  		#get current axis
ax1.spines['top'].set_color('none')		#get rid of black border line
ax1.spines['bottom'].set_color('none')
ax1.spines['left'].set_color('none')		
ax1.spines['right'].set_color('none')
ax1.xaxis.set_ticks_position('bottom')
ax1.spines['bottom'].set_position(('data',0))
ax1.spines['bottom'].set_color('gray')
ax1.yaxis.set_ticks_position('left')
ax1.spines['left'].set_position(('data',0))
ax1.spines['left'].set_color('gray')

#annotating a specific plot point
i = -plb.pi/2
plb.plot([i, i], [0, plb.sin(i)], color = 'cyan', linewidth=1.25, linestyle='-.')
plb.scatter([i, ],[plb.sin(i), ], 25, color ='red')
plb.annotate(r'$sin(-\frac{pi}{2})=-1$', xy=(i, plb.sin(i)), xycoords='data', textcoords='offset points', xytext=(-25, +75), fontsize=16, color='brown', arrowprops=dict(arrowstyle="-|>", color='brown', connectionstyle="arc3,rad=.65"))

#other finer touches
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
	label.set_fontsize(12)
	label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.85))
	
#change figure name
fig = plb.gcf()
fig.canvas.set_window_title('sin and cos')

#plot titles
plb.title('Plot of the Sin and Cos Functions')
plb.xlabel('period, rd', fontsize=9, position=(0.065,0), rotation=5, color='gray', alpha=0.75)
plb.ylabel('amplitude', fontsize=9, position=(0, 0.75), color='gray', aplha=.75)
plb.grid()    		# place a grid

plb.hold(True)		#hold so that another plot can be drawn on top of the current

#closes figure after 5 second pause
plb.pause(5)
plb.close
string = ('Here is some description to the plot')
plb.text(-5,-8.5.,string)

"""
Homework # 3
Part 1 – create your data:
1. Include a sec4on line with your name
2. Work only with these imports:
from numpy import matrix, array, random, min, max
import pylab as plb (… or use matplotlib)
3. Cerate a list A of 600 random numbers bound between (0:10)
4. Create an array B with with 500 elements bound in the range [-3*pi:2*pi]
5. Using if, for or while, create a func4on that overwrites every element in A that falls outside of the interval [2:9), and overwrite
that element with the average between the smallest and largest element in A
6. Normalize each list element to be bound between [0:0.1]
7. Return the result from the func4on to C
8. Cast C as an array
9. Add C to B (think of C as noise) and record the result in D … (watch out: C is of different length. Truncate it)
Part 2 - ploXng:
10. Create a figure, give it a 4tle and specify your own size and dpi
11. Plot the sin of D, in the (2,1,1) loca4on of the figure
12. Overlay a plot of cos using D, with different color, thickness and type of line
13. Create some space on top and boZom of the plot (on the y axis) and show the grid
14. Specify the following: 4tle, Y-axis label and legend to fit in the best way
15. Plot the tan of D, in loca4on (2,1,2) with grid showing, X-axis label, Y-axis label and legend on top right
16. Organize your code: use each line from this HW as a comment line before coding each step
17. Save these steps in a .py file and email it to me before next class. I will run it!
"""

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

"""
LECTURE 

ADVANCED PLOTTING
"""

#Bar plot
import pylab as plb
k = 8
x = plb.arange(k)
y1 = plb.rand(k)*(1-x/k)
y2 = plb.rand(k) * (1-x/k)
plb.axes([0.075, 0.075, .88, .88])

plb.bar(x, +y1, facecolor='#9922aa', edgecolor='green')
plb.bar(x, -y2, facecolor='#ff3366', edgecolor='green')

for a, b in zip(x, y1):
	plb.text(a+0.41, b+0.08, '%.3f' % b, ha='center', va='bottom')
for a, b in zip(x, y2):
	plb.text(a+0.41, -b-0.08, '%.3f' % b, ha='center', va='bottom')	

plb.xlim(-.5, k), plb.ylim(-1.12, +1.12)
plb.grid(True)
plb.show()

# Scatter plot
x = plb.rand(1,2,1500)
y = plb.rand(1,2,1500)
plb.axes([0.075, 0.075, .88, .88])

plb.cla()
plb.scatter(x, y, s=65, alpha=.75, linewidth=.125, c=plb.arctan2(x,y))

plb.grid(True)
plb.xlim(-0.085,1.085), plb.ylim(-0.085,1.085)
plb.pause(1)

#image plot
plb.cla()
array = plb.random((80,120))
plb.imshow(array, cmap=plb.cm.gist_rainbow) # with specific colormap
plb.pause(1))

import matplotlib.image as img
import matplotlib.pyplot as plt
image = img.read('files/lecture5/file.jpg')
plt.imshow(image)

# luminosity display using 1-channel only(no RGB color)
# a default colormap (lookuplabel - LUT) is applied called 'jet':

luminosity = image[:,:,0]
plt.imshow(luminosity)
plt.pause(5)

#other colormaps:
plt.imshow(luminosity, cmap='hot')
plt.pause(5)
plt.imshow(luminosity, cmap='spectral')plt.pause(5)

# histogram 
plb.figure(1)
gaus_dist = plb.normal(-2,2,size=512)  #create a random floating point vector
plb.hist(gaus_dist, normed=True, bins=24)  #default: bins=10, color= 'blue'

plb.title("Gaussian distribution/ Histogram")
plb.xlabel('Value')
plb.ylabel('Frequency')
plb.grid(True)
plb.show()

plb.figure(2)
gaus_dist = plb.normal(size=512)
unif_dist = plb.uniform(-5,5,size=512)   #create uniform distribution vector

plb.hist(unif_dist, bins=24, histtype='stepfilled', normed=True, color='cyan', label='Uniform')
plb.hist(gaus_dist, bins=24, histtype='stepfilled', normed=True, color='orange', label='Gaussian', alpha=0.065)

plb.legend(loc='upper left')
plb.title('Gaussian vs Uniform distribution/Histogram')
plb.xlabel('Value')
plb.ylabel('Frequency')
plb.grid(True)
plb.show()

# Pie Chart
plb.figure('How do we get to work:')
plb.axes([0.035, 0.035, 0.9, 0.9])
l = 'Car', 'Truck', 'Boat', 'Dingie', 'Train', 'Plane', 'Bus', 'Rocket', 'Tram', 'Other'
b = plb.round_(plb.random(10), decimals=2)
c = ['blue', 'red', 'green', 'gray', 'yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'cyan', 'orange']
e = (0, 0, 0, 0, 0, 0, 0, 0.05, 0, 0) #'explode' the 8th slice only

plb.cla()
plb.pie(b, explode = e, labels=l, colors=c, radius=.75, autopct='%1.2f%%', shadow=True, startangle=15)

#we set aspect ratio to 'equal' so the pie is drawn in a circle
plb.axis('equal')
plb.xticks(()); plb.yticks(())
plb.show()

# Contour plot:

def f(x,y):
	return (2- x/3 + x**6 +2.125*y) * plb.exp(-x**2-y**2)

n = 128
x = plb.linspace(-2, 2, n)
y = plb.linspace(-1, 1, n)
A,B = plb.meshgrid(x, y)

plb.cla()
plb.axes([0.075, 0.075, 0.92, 0.92])

plb.countourf(A, B, f(A,B), 12, alpha=.5, cmpa = plb.cm.gist_rainbow)
c = plb.contour(A, B, f(A,B), 8, colors = 'black', linewidth=.65)

plb.clabel(C, inline=1, fontsize=14)
plb.xticks(()); plb.yticks(())
plb.show()

# Polar Plot
 a = plb.axes([0.065, 0.065, 0.88, 0.88], polar=True)

q=24
t=plb.arange(0.015, 3*plb.pi, 3*plb.pi/q)
rad = 12 * plb.rand(q)
w = plb.pi/4 * plb.rand(q)
ba = plb.bar(t, rad, width = w)

for r,bar in zip(rad, ba):
	bar.set_facecolor(plb.cm.jet(r/12.))
	bar.set_alpha(0.75)
	
plb.show()

# 3 dimensional plotting
from mpl_toolkits.mplot3d import Axes3D

ax = Axes3D(plb.figure())
x = plb.arange(-6, 3, 0.35)
y = plb.arange(-6, 6, 0.35)
x, y = plb.meshgrid(x, y)
k = plb.sqrt(x**2 + y**2)
z = plb.sin(k)

ax.plot_surface(x, y, z, rstride=2, cstride=1, cmap=plb.cm.gist_rainbow)
ax.contourf(x, y, z, zdir='z', offset=-3, cmap=plb.cm.gist_stern)
ax.set_zlim(-4, 4)

plb.show()

###
### ARRAY OPERATIONS
###
import numpy as np
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
# Elementwise sum; both produce the array # [[ 6.0 8.0] # [10.0 12.0]]
print(x + y)
print(np.add(x, y))
# Elementwise difference; both produce the array # [[-4.0 -4.0] # [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))
# Elementwise product; both produce the array # [[ 5.0 12.0] # [21.0 32.0]]
print(x * y)
print(np.mul4ply(x, y))
# Elementwise division; both produce the array # [[ 0.2 0.33333333] # [ 0.42857143 0.5 ]]
print(x / y)
print(np.divide(x, y))
# Elementwise square root; produces the array # [[ 1. 1.41421356] # [ 1.73205081 2. ]]
print(np.sqrt(x))


m = plb.zeros((3,), dtype=[('Ticker:', 'S4'),()



