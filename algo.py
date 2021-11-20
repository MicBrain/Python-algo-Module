###########################
########## TREES ##########
###########################

""" 1. GENERAL TREE

DEFINITION:
	A Tree is a non-linear abstract data type or data structure that simulates a 
hierarchical tree structure, with a root value and subtrees of children, represented
as a set of linked nodes.
			    34
			   /  \
			 23    43
			/  \  /  \ 
		      12   5  2	  324
"""

class Tree(object):
    """ A tree with internal values. """

    def __init__(self, entry, left=None, right=None):
        self.entry = entry
        self.left = left
        self.right = right

    def __repr__(self):
        args = repr(self.entry)
        if self.left or self.right:
            args += ', {0}, {1}'.format(repr(self.left), repr(self.right))
        return 'Tree({0})'.format(args)

    def tree_print(self):
        def print_helper(tree, depth):
            if tree.right:
                print_helper(tree.right, depth + 1)
            print("{0}{1}".format("\t" * depth, tree.entry))
            if tree.left:
                print_helper(tree.left, depth + 1)
        print_helper(self, 0)

def swap_tree(tree):
    """ Swaps the left and right branches of a tree. """
    if tree is None:
        return
    tree.left, tree.right = tree.right, tree.left
    swap_tree(tree.left)
    swap_tree(tree.right)

def traversal(tree):
	""" Traverses the tree in order to get all elements in some sort of list. """
	if tree is None:
		return
	traversal(tree.left)
	print(tree.entry)
	traversal(tree.right)

def tree_to_list(tree):
	""" Saves all data in tree in the list. """
	lst = []
	if tree is not None:
		if tree.right:
			lst.extend(tree_to_list(tree.right))
		lst.append(tree.entry)
		if tree.left:
			lst.extend(tree_to_list(tree.left))
	return lst

def equal_trees(t1, t2):
	""" Checks if both trees are equal. """
	if t1 is None and t2 is None:
		return True
	elif t1 is None or t2 is None:
		return False
	elif t1.entry != t2.entry:
		return False
	else:
		return equal_trees(t1.left, t2.left) and equal_trees(t1.right, t2.right)

def tree_size(t):
	""" Returns the number of elements in a tree. """
	if t is None:
		return 0
	else:
		return 1 + tree_size(t.left) + tree_size(t.right)

def tree_height(t):
	""" Returns the height of a tree. """
	if t is None or not t.left and not t.right:
		return 0
	else:
		return 1 + max(tree_height(t.left), tree_height(t.right))

"""  2. BINARY TREE

DEFINITION:
    A binary search tree (BST) is a node-based binary tree data structure which has the following 
properties:
		1. The left subtree of a node contains only nodes with keys less than the node’s key.
		2. The right subtree of a node contains only nodes with keys greater than the node’s key.
		3. Both the left and right subtrees must also be binary search trees.
					    8
					   / \
					  4   100
					 / \   / \
					2  5  7   234
"""

class Binary_Tree(object):
	""" Node class contains left and right child, and a data."""
	
	def __init__(self, data = None):
		""" Binary_Tree Constructor. """
		self.left = None
		self.right = None
		self.data = data

	def insert(self, data):
		""" Insert new node with data. """
		if data > self.data:
			if self.right is None:
				self.right = Binary_Tree(data)
			else:
				self.right.insert(data)
		elif data < self.data:
			if self.left is None:
				self.left = Binary_Tree(data)
			else:
				self.left.insert(data)

	def lookup(self, data, parent = None):
		""" Lookup node containing data. """
		if data < self.data:
			if self.left is None:
				return None, None
			return self.left.lookup(data, self)
		elif data > self.data:
			if self.right is None:
				return None, None
			return self.right.lookup(data, self)
		else:
			return self, parent

	def delete(self, data):
		""" Delete node containing data. """
		node, parent = self.lookup(data)
		if node is not None:
			children_count = node.children_count()
			if children_count == 0:
				# if node has no children, just remove it
				if parent:
					if parent.left is node:
						parent.left = None
					else:
						parent.right = None
				del node
			elif children_count == 1:
				# if node has 1 child replace node by its child
				if node.left:
					item = node.left
				else:
					item = node.right
				if parent:
					if parent.left is node:
						parent.left = item
					else:
						parent.right = item
				del node
			else:
				# if node has 2 children find its successor
				parent = node
				successor = node.right
				while successor.left:
					parent = successor
					successor = successor.left
				node.data = successor.data
				if parent.left == successor:
					parent.left = successor.right
				else:
					parent.right = successor.right

	def children_count(self):
		cnt = 0
		if self.left:
			cnt += 1
		if self.right:
			cnt += 1
		return cnt

	def print_tree(self):
		""" Print tree content. """
		if self.left:
			self.left.print_tree()
		print (self.data)
		if self.right:
			self.right.print_tree()

	def compare_trees(self, node):
		""" Compares 2 trees. """
		if node is None:
			return False
		if self.data != node.data:
			return False
		result = True
		if self.left is None:
			if node.left:
				return False
		else:
			result = self.left.compare_trees(node.left)
		if result is False:
			return False
		if self.right is None:
			if node.right:
				return False
		else:
			result = self.right.compare_trees(node.right)
		return result

	def tree_data(self):
		""" Generator to get the tree nodes data. """
		final_list = []
		node = self
		while final_list or node:
			if node:
				final_list.append(node)
				node = node.left
			else:
				node = final_list.pop()
				yield node.data
				node = node.right

############################
#### SORTING ALGORITHMS ####
############################

"""  3. BUBBLE SORT

DEFINITION
	The bubble sort makes multiple passes through a list. It compares adjacent items and 
exchanges those that are out of order. Each pass through the list places the next largest 
value in its proper place. In essence, each item “bubbles” up to the location where it 
belongs.
"""

def Bubble_Sort(lst):
    for item in range(len(lst)-1,0,-1):
        for i in range(item):
            if lst[i]>lst[i+1]:
                temporary = lst[i]
                lst[i] = lst[i+1]
                lst[i+1] = temporary

def Short_Bubble(lst):
    exchanges = True
    item = len(lst)-1
    while item > 0 and exchanges:
       exchanges = False
       for i in range(item):
           if lst[i]>lst[i+1]:
               exchanges = True
               temporary = lst[i]
               lst[i] = lst[i+1]
               lst[i+1] = temporary
       item = item-1

"""  4. SELECTION SORT

DEFINITION
	The selection sort improves on the bubble sort by making only one exchange for every pass
through the list.
"""

def Selection_Sort(lst):
   for fillslot in range(len(lst)-1,0,-1):
       Max_Position=0
       for place in range(1,fillslot+1):
           if lst[place]>lst[Max_Position]:
               Max_Position = place
       temp = lst[fillslot]
       lst[fillslot] = lst[Max_Position]
       lst[Max_Position] = temp

"""  5. INSERTION SORT

DEFINITION
	The insertion sort always maintains a sorted sublist in the lower positions of the list.
Each new item is then “inserted” back into the previous sublist such that the sorted sublist
is one item larger.
"""

def Insertion_Sort(lst):
   for index in range(1,len(lst)):
     current = lst[index]
     position = index
     while position>0 and lst[position-1]>current:
         lst[position]=lst[position-1]
         position = position-1
     lst[position]=current

"""  6. SHELL SORT

DEFINITION
	The shell sort improves on the insertion sort by breaking the original list into a number
of smaller sublists, each of which is sorted using an insertion sort.
"""

def Shell_Sort(lst):
    sublistcount = len(lst)//2
    while sublistcount > 0:
      for startposition in range(sublistcount):
        Insertsort(lst,startposition,sublistcount)
      sublistcount = sublistcount // 2

def Insertsort(lst,start,gap):
    for i in range(start+gap,len(lst),gap):
        currentvalue = lst[i]
        position = i
        while position>=gap and lst[position-gap]>currentvalue:
            lst[position]=lst[position-gap]
            position = position-gap
        lst[position]=currentvalue

"""  7. MERGE SORT

DEFINITION
	Merge sort is a recursive algorithm that continually splits a list in half. If the list is
empty or has one item, it is sorted by definition (the base case). If the list has more than
one item, we split the list and recursively invoke a merge sort on both halves. Once the two 
halves are sorted, the fundamental operation, called a merge, is performed.
"""

def Merge_Sort(lst):
    if len(lst)>1:
        mid = len(lst)//2
        left = lst[:mid]
        right = lst[mid:]
        Merge_Sort(left)
        Merge_Sort(right)
        i=0
        j=0
        k=0
        while i<len(left) and j<len(right):
            if left[i]<right[j]:
                lst[k]=left[i]
                i=i+1
            else:
                lst[k]=right[j]
                j=j+1
            k=k+1
        while i<len(left):
            lst[k]=left[i]
            i=i+1
            k=k+1
        while j<len(right):
            lst[k]=right[j]
            j=j+1
            k=k+1

"""  8. QUICK SORT

DEFINITION
	The quick sort uses divide and conquer to gain the same advantages as the merge sort, 
while not using additional storage. As a trade-off, however, it is possible that the list 
may not be divided in half. When this happens, we will see that performance is diminished.
"""

def Quick_Sort(lst):
	def partition(lst,first,last):
		central = lst[first]
		left = first+1
		right = last
		boolean = False
		while not boolean:
			while left <= right and lst[left] <= central:
				left = left + 1
			while lst[right] >= central and right >= left:
				right = right -1
			if right < left:
				boolean = True
			else:
				temporary = lst[left]
				lst[left] = lst[right]
				lst[right] = temporary
		temporary = lst[first]
		lst[first] = lst[right]
		lst[right] = temporary
		return right
	def sort_helper(lst,first,last):
		if first<last:
			splitpoint = partition(lst,first,last)
			sort_helper(lst,first,splitpoint-1)
			sort_helper(lst,splitpoint+1,last)
	sort_helper(lst,0,len(lst)-1)

##########################
### HASHING ALGORITHMS ###
##########################

"""  9. STRING HASHING FUNCTION

DEFINITION
	Hash values are integers. They are used to quickly compare dictionary keys during a dictionary 
lookup. Numeric values that compare equal have the same hash value.
"""

def Hash(s, v = 2):
	""" Hashing Function"""
	T = 10000000 # Hashing Function should be effective for all table sizes.
	v = 7 
	def init(v):
		return v
	def step(i, h, c):
		return h ^ ((h << 5) + (h << 2) + c)
	def final(h, v):
		return h % T
	current = init(v)
	for i in range(1, len(s)):
		char = s[i]
		current = step(i, current, ord(str(char)))
	return final(current, v)

"""  10. FAST HASHING FUNCTION

DEFINITION
	Very similar to the String Hashing Function. Basically uses the same algorithm. However,
due to some changes works faster.
"""

def Fast_Hash(s, v = 2):
    """ Fast Hashing Function"""
    T = 10000000 # Hashing Function should be effective for all table sizes.
    v = 7 
    def step(i, h, c):
    	return h ^ ((h << 5) + (h << 2) + c)
    def final(h, v):
    	return h % T
    current = ord(s[0])
    if len(s) == 1:
        return current
    for i in range(0, len(s)//2):
        char = s[i]
        current = step(i, current, ord(str(char)))
    return final(current, v)

###########################
###### NUMBER THEORY ######
###########################

"""  11. PRIME CHECKER

DEFINITION
	A prime number (or a prime) is a natural number greater than 1 that has no positive divisors
other than 1 and itself. This function check if the given number is a prime number or not.
"""
def is_prime(num):
	""" Checks if the given number is a prime number or not. """
	if num > 1:
		for i in range(2, num):
			if (num % i) == 0:
				return False
		else:
			return True
	else:
		return False

"""  12. SIEVE OF ERATOSTHENES                  

DEFINITION
	Sieve of Eratosthenes is an algorithm that gives the list of all prime numbers in a given
region.
"""
def primes(limit):
    if limit < 2: return []
    if limit < 3: return [2]
    lmtbf = (limit - 3) // 2
    buf = [True] * (lmtbf + 1)
    for i in range((int(limit ** 0.5) - 3) // 2 + 1):
        if buf[i]:
            p = i + i + 3
            s = p * (i + 1) + i
            buf[s::p] = [False] * ((lmtbf - s) // p + 1)
    return [2] + [i + i + 3 for i, v in enumerate(buf) if v]

"""  13. PRIME FACTORIZATION                 

DEFINITION
	This function returns a list of all prime factors of the number.
"""
def Prime_Factorization(x):
	""" Gives the list of all prime factors of the number. """
	lst=[]
	cycle=2
	while cycle<=x:
		if x%cycle==0:
			x/=cycle
			lst.append(cycle)
		else:
			cycle+=1
	return lst

"""  14. FIBONACCI NUMBERS                 

DEFINITION
	In mathematics, the Fibonacci numbers or Fibonacci sequence are the numbers in the 
following integer sequence:
 					1, 1, 2, 3, 4, 8, 13, 21, 34, , 55, 89, ...
 	By definition, the first two numbers in the Fibonacci sequence are 1 and 1, or 0 and 1, 
depending on the chosen starting point of the sequence, and each subsequent number is the sum
of the previous two. 
"""
def fib(n):
	""" This function generates the nth Fibonacci number in a fast way. """
	def fib(prvprv, prv, c):
		if c < 1:
			return prvprv
		else: 
			return fib(prv, prvprv + prv, c - 1) 
	return fib(0, 1, n)

"""  15. LEAST COMMON MULTIPLE               

DEFINITION
	In arithmetic and number theory, the least common multiple (also called the lowest common
multiple or smallest common multiple) of two integers a and b, usually denoted by LCM(a, b), 
is the smallest positive integer that is divisible by both a and b.
"""

def LCM(a, b):
    """ Implementation of of LCM algorithm. """
    temporary = a
    while (temporary % b) != 0:
        temporary += a
    return temporary

"""  16. NUMBER OF TOTATIVES OF A NUMBER        

DEFINITION
	In number theory, a totative of a given positive integer n is an integer k such that 
0 < k < n and k is coprime to n.
"""

def Alg_improved_Euclid( a, b ):
    """ Simple recursive implementation (for gcd) """
    if a < b: a, b = b, a
    if a % b == 0: return b
    #print a, b
    if b == 0: return a
    else: return Alg_improved_Euclid( b, a % b )

def Count_Totatives(n): 
    """ Counts totatives of a number. """ 
    if not (type(n)==type(1) and n>=0): 
    	raise ValueError('Invalid input type')
    tot,pos = 1, n-1   
    while pos>1: 
       if Alg_improved_Euclid(pos,n)==1: tot += 1 
       pos -= 1 
    return tot

"""  17. GCD       

DEFINITION
	In mathematics, the greatest common divisor (gcd), is the largest positive integer that
divides the numbers without a remainder.
"""

def GCD(a,b):
	""" Calculates the GCD of two numbers. """
	try:
		aInt = isinstance(a,int)
		bInt = isinstance(b,int)
		if not (aInt and bInt):
			raise TypeError("Argument of GCD is not integer.")
	except:
		raise
	else:
		if a==0:
			return b
		if b==0:
			return a
		A = max(abs(a),abs(b))
		B = min(abs(a),abs(b))
		while B!=0:
			tmp = B
			B = A%B
			A = tmp
		return A

"""  18. FACTORS OF A NUMBER       

DEFINITION
	In mathematics a divisor of an integer n, also called a factor of n, is an integer that 
can be multiplied by some other integer to produce n. This function returns a dictionary with
factors of the number as values and the number of their occurrences as keys of the dictionary.
"""
def Factor(n):
	""" Provides all the factors of the number. """
	try:
		if not isinstance(n,int):
			raise TypeError('Argument of Factor(n) is not integer')
	except:
		raise
	else:
		factors = {}
		if n<0:
			n = -n
			factors[-1]=1
		if n==0 or n==1:
			return {n:1}
		mid = int(n**(1/2.0))
		n_copy = n
		for i in range(2,mid+1):
			k = 0
			if n_copy%i==0:
				while (n_copy%i==0):
					k = k+1
					n_copy = n_copy/i
				factors[i] = k
		if n_copy>1:
			factors[n_copy] = 1			
		return factors

"""  19. PELL'S EQUATION       

DEFINITION
	Pell's equation is any Diophantine equation of the form x^2 - n*(y^2) = 1 where n is a 
given nonsquare integer and integer solutions are sought for x and y. This algorithm returns
a list with the first k solutions of the Pell's equation.
"""

from math import sqrt
def Pell_Equations(n, k):
    sols = []
    r = sqrt(n)
    a = int(r)
    p0, p1 = 1, a
    q0, q1 = 0, 1
    while p1*p1 - n*q1*q1 != 1:
        r = 1.0 / (r - a)
        a = int(r)
        p0, p1 = p1, a*p1 + p0
        q0, q1 = q1, a*q1 + q0
    f = f0 = p1+q1*sqrt(n)
    g = g0 = p1-q1*sqrt(n)
    for k in range(1,k+1):
        x = int((f + g)/2 + 0.0001)
        y = int((f - g)/(2*sqrt(n)) + 0.0001)
        f *= f0
        g *= g0
        sols.append([x, y])
    return sols

"""  20. PERFECT NUMBERS   

DEFINITION
	In number theory, a perfect number is a positive integer that is equal to the sum of its 
proper positive divisors, that is, the sum of its positive divisors excluding the number itself
(also known as its aliquot sum). 
"""
def perfect(n):
    sum = 0
    for i in range(1, n):
        if n % i == 0:
            sum += i
    return sum == n

############################
### SEARCHING ALGORITHMS ###
############################

"""  21. BINARY SEARCH                 

DEFINITION
	In computer science, a binary search or half-interval search algorithm finds the position
of a specified input value (the search "key") within an array sorted by key value.
"""

def Binary_Search(l, value, low = 0, high = -1):
    if not l: return -1
    if(high == -1): high = len(l)-1
    if low == high:
        if l[low] == value: return low
        else: return -1
    mid = (low+high)//2
    if l[mid] > value: return Binary_Search(l, value, low, mid-1)
    elif l[mid] < value: return Binary_Search(l, value, mid+1, high)
    else: return mid

"""  22. DEPTH-FIRST SEARCH                 

DEFINITION
	Depth-first search (DFS) is an algorithm for traversing or searching tree or graph data 
structures. One starts at the root (selecting some arbitrary node as the root in the case of 
a graph) and explores as far as possible along each branch before backtracking.
"""  
def DFS(graph, start, path = []):
	""" Implementation of DFS algorithm. """
	if start not in graph or graph[start] == None or graph[start] == []:
		return None
	path = path + [start]
	for edge in graph[start]:
		if edge not in path:
			path = DFS(graph, edge,path)
	return path

"""  23. KNUTH-MORRIS-PRATT ALGORITHM              

DEFINITION
	In computer science, the Knuth–Morris–Pratt string searching algorithm (or KMP algorithm)
searches for occurrences of a "word" W within a main "text string" S by employing the observation
that when a mismatch occurs, the word itself embodies sufficient information to determine where
the next match could begin, thus bypassing re-examination of previously matched characters.
"""

def KMP(string, word):
    word_length = len(word)
    string_length = len(string)
    offsets = []

    if word_length > string_length:
        return offsets

    prefix = compute_prefix(word)
    q = 0
    for index, letter in enumerate(string):
        while q > 0 and word[q] != letter:
            q = prefix[q - 1]
        if word[q] == letter:
            q += 1
        if q == word_length:
            offsets.append(index - word_length + 1)
            q = prefix[q - 1]
    return offsets

def compute_prefix(word):
    word_length = len(word)
    prefix = [0] * word_length
    k = 0

    for q in range(1, word_length):
        while k > 0 and word[k] != word[q]:
            k = prefix[k - 1]

        if word[k + 1] == word[q]:
            k = k + 1
        prefix[q] = k
    return prefix

"""  24. BOYER-MOORE-HORSPOOL ALGORITHM             

DEFINITION
	In computer science, the Boyer–Moore–Horspool algorithm or Horspool's algorithm is an algorithm
for finding substrings in strings. 
"""

def BMH(text, pattern):
    pattern_length = len(pattern)
    text_length = len(text)
    offsets = []
    if pattern_length > text_length:
        return offsets
    bmbc = [pattern_length] * 256
    for index, char in enumerate(pattern[:-1]):
        bmbc[ord(char)] = pattern_length - index - 1
    bmbc = tuple(bmbc)
    search_index = pattern_length - 1
    while search_index < text_length:
        pattern_index = pattern_length - 1
        text_index = search_index
        while text_index >= 0 and \
                text[text_index] == pattern[pattern_index]:
            pattern_index -= 1
            text_index -= 1
        if pattern_index == -1:
            offsets.append(text_index + 1)
        search_index += bmbc[ord(text[search_index])]
    return offsets


"""  25. RABIN-KARP SEARCH ALGORITHM             

DEFINITION
	In computer science, the Rabin–Karp algorithm or Karp–Rabin algorithm is a string searching
algorithm created by Richard M. Karp and Michael O. Rabin (1987) that uses hashing to find any 
one of a set of pattern strings in a text.
"""

class RollingHash:
	def __init__(self, string, size):
		self.str  = string
		self.hash = 0
		for i in range(0, size):
			self.hash += ord(self.str[i])
		self.init = 0
		self.end  = size

	def update(self):
		if self.end <= len(self.str) -1:
			self.hash -= ord(self.str[self.init])
			self.hash += ord(self.str[self.end])
			self.init += 1
			self.end  += 1

	def digest(self):
		return self.hash

	def text(self):
		return self.str[self.init:self.end]

def RKS(substring, string):
	if substring == None or string == None:
		return -1
	if substring == "" or string == "":
		return -1
	if len(substring) > len(string):
		return -1
	hs 	 = RollingHash(string, len(substring))
	hsub = RollingHash(substring, len(substring))
	hsub.update()
	for i in range(len(string)-len(substring)+1):						
		if hs.digest() == hsub.digest():
			if hs.text() == substring:
				return i
		hs.update()
	return -1

#######################
### DATA ALGORITHMS ###
#######################

"""  26. SHUFFLING ALGORITHM            

DEFINITION
	This algorithm shuffles all the items in the list.
"""
from random import seed, randint

def shuffle(seq):
    seed()
    for i in reversed(range(len(seq))):
        j = randint(0, i)
        seq[i], seq[j] = seq[j], seq[i]
    return seq

"""  27. JACCARD INDEX           

DEFINITION
	The Jaccard index, also known as the Jaccard similarity coefficient (originally coined
coefficient de communauté by Paul Jaccard), is a statistic used for comparing the similarity 
and diversity of sample sets.
"""

def jaccard_index(dataset_1, dataset_2):
	length = len(set(dataset_1).intersection(set (dataset_2)))
	return  length / float(len(dataset_1) + len(dataset_2) - length)

def jaccard_distance(dataset_1, dataset_2):
	return 1 - jaccard_index(dataset_1, dataset_2)

def percentage_of_similarity(dataset_1, dataset_2):
	return float(format(100 * jaccard_index(dataset_1, dataset_2), '.3f'))

"""  28. LEVENSHTEIN DISTANCE            

DEFINITION
	In information theory and computer science, the Levenshtein distance is a string metric for
measuring the difference between two sequences.
"""

def Levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return Levenshtein_distance(s2, s1)
    if not len(s2):
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i+1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j+1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return current_row[-1]

"""  29. MAXIMUM SUBARRAY PROBLEM            

DEFINITION
	The maximum subarray problem is the task of finding the contiguous subarray within a one-dimensional
array of numbers which has the largest sum. Kadane's algorithm finds the maximum subarray sum 
in linear time.
"""

def find_max_subarray(numbers):
    max_till_here = [0]*len(numbers)
    max_value = 0
    for i in range(len(numbers)):
        max_till_here[i] = max(numbers[i], max_till_here[i-1] + numbers[i])
        max_value = max(max_value, max_till_here[i])
    return max_value

"""  30. LIST PERMUTATIONS            

DEFINITION
	This generator returns all the permutations for the given list.
"""
def permutations(l):
    """ Generator for list permutations. """
    if len(l) <= 1:
        yield l
    else:
        a = [l.pop(0)]
        for p in permutations(l):
            for i in range(len(p)+1):
                yield p[:i] + a + p[i:]

"""  31. PALINDROM STRING           

DEFINITION
	A palindrome is a word, phrase, number, or other sequence of symbols or elements that reads 
the same forward or reversed, with general allowances for adjustments to punctuation and word
dividers.
"""
def is_polindrom(string):
	""" This function checks whether the given string is a polindrom or not. """
	for i,char in enumerate(string):
		if char != string[-i-1]:
			return False
	return True

"""  32. THE MOST COMMON ELEMENT IN THE LIST          

DEFINITION
	This function helps to identify the most common element in the given list.
"""

def Most_Common(lst):
	from collections import Counter
	data = Counter(lst)
	return data.most_common(1)[0][0]

"""  33. ANAGRAMS          

DEFINITION
	This function checks if two strings are anagrams
"""

def is_Anagram(s1,s2):
	if len(s1) != len(s2):
		return False
	s1 = list(s1).sort()
	s2 = list(s2).sort()
	if(s1 == s2):
		return True
	else:
		return False

"""  34. FIND NONREPEARED CHARACTER         

DEFINITION
	This function finds the first nonrepeated character in the string.
"""
def Find_Nonrepeat(s):
	d = dict()
	for l in s:
		if l in d.keys():
			d[l] += 1
		else:
			d[l] = 1
	for l in s:
		if d[l] == 1:
			return l

"""  35. ATOI        

DEFINITION
	The atoi function converts str into an integer, and returns that integer.
"""

def atoi(str):
	maxTenPower = len(str)
	if '-' == str[0]:
		negative = True
		start = 1
	else:
		negative = False
		start = 0
	sum = 0
	ord0 = ord('0')
	for i in range(start, maxTenPower):
		sum += pow(10, i-start) * (ord(str[maxTenPower-1-i])-ord0)
	if negative:
		return (-1*sum)
	else:
		return sum

"""  36. WEIGHTED RANDOM        

DEFINITION
	This function generate random integers given a list of integer weights.
"""
import random
def Weighted_Random(W):
    r = random.random()
    weight_sum = float(sum(W))
    last_theta = 0
    for i, weight in enumerate(W):
        theta = last_theta + (weight / weight_sum)
        if r <= theta:
            return i
        last_theta = theta

"""  37. HAMMING DISTANCE

DEFINITION
	In information theory, the Hamming distance between two strings of equal length
is the number of positions at which the corresponding symbols are different.
"""
def Hamming_Distance(str1, str2):
	diffs = 0
	for ch1, ch2 in zip(str1, str2):
		if ch1 != ch2:
			diffs += 1
	return diffs
