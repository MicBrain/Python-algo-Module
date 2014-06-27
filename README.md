# algo -- Module for Algorithms
============================================================================================================================

Source code: algo.py

   This is the Official Documentation of algo module. This module provides an implementation of different algorithms. The module was created in order to help Python developers easily implement some frequently used algorithms. Written algorithms include topics from different subfields of Computer Science. The current version of the module contains 50 algorithms. It is important to note that the license of algo module is distributed under Apache Licese.


## Installation

   In order to fully use the algo module firstly download the whole package from the following link: https://github.com/Rafa1994/Python-algo-Module . Then open the terminal and go to the directory where the module package is located. After that run this command from terminal:
###### python setup.py sdist
   For Windows, open a command prompt windows (Start ‣ Accessories) and change the command to:
###### setup.py sdist
   Finally in the same package directory type this following command:
###### python setup.py install


## Module Content

#### TREE STRUCTURES

##### a) Trees
Tree(entry, left = None, right = None) : -  Creates a General Structure for Tree. A Tree is a non-linear abstract data type or data structure that simulates a hierarchical tree structure, with a root value and subtrees of children, represented as a set of linked nodes.

   1. Tree method: tree_print() - Gives the visualised structure of the tree created by Tree class.
    
   2. Function: swap_tree(tree) - Swaps the left and right branches of a tree.
   
   3. Function: traversal(tree) - Traverses the tree in order to get all elements in some sort of list.
   
   4. Function: tree_to_list(tree) - Saves all data in tree in the list.
   
   5. Function: equal_trees(t1, t2) - Checks if trees t1 and t2 are equal.
   
   6. Function: tree_size(t) - Returns the number of elements in a tree t. 
   
   7. Function: tree_height(t) - Returns the height of a tree t.

##### b) Binary Trees
Binary_Tree(data = None), self.left = None, self.right = None : - Creates a structure for Binary Tree. A binary search tree (BST) is a node-based binary tree data structure which has the following properties: a) The left subtree of a node contains only nodes with keys less than the node’s key, b) The right subtree of a node contains only nodes with keys greater than the node’s key, c) Both the left and right subtrees must also be binary search trees.

   1. Binary_Tree method: insert(self, data) - Inserts new node with data.
    
   2. Binary_Tree method: lookup(self, data, parent = None) - Looks up a node containing data and returns also its parent.

   3. Binary_Tree method: delete(self, data) - Deletes a node containing data from binary tree. 

   4. Binary_Tree method: children_count(self) - Counts the number of nodes in a tree. 
   
   5. Binary_Tree method: print_tree(self) - Prints the content of the Binary tree.
   
   6. Binary_Tree method: compare_trees(self, node) -  Compates two binary trees.
   
   7. Binary_Tree method: tree_data(self) - A generator that gets a binary tree node data.

#### SORTING ALGORITHMS
A sorting algorithm is an algorithm that puts elements of a list(lst) in a certain order. The most-used orders are numerical order and lexicographical order. Efficient sorting is important for optimizing the use of other algorithms (such as search and merge algorithms) which require input data to be in sorted lists(lst).

   1. Bubble_Sort(lst) - For description read: http://en.wikipedia.org/wiki/Bubble_sort
   
   2. Short_Bubble(lst) - Faster version of Bubble Sort.
   
   3. Selection_Sort(lst) - For description read: http://en.wikipedia.org/wiki/Selection_sort
   
   4. Insertion_Sort(lst) - For description read: http://en.wikipedia.org/wiki/Insertion_sort
   
   5. Shell_Sort(lst) - For description read: http://en.wikipedia.org/wiki/Shellsort
   
   6. Merge_Sort(lst) - For description read: http://en.wikipedia.org/wiki/Merge_sort
   
   7. Quick_Sort(lst) - For descritopn read: http://en.wikipedia.org/wiki/Quicksort
   

#### HASHING ALGORITHMS
Hash values are integers. They are used to quickly compare dictionary keys during a dictionary lookup. Numeric values that compare equal have the same hash value.
   
   1. Hash(string) - Hashing function that converts a string value to an integer value.
   
   2. Fast_Hash(string) - A faster version of Hash(string) function.
   

#### NUMBER THEORY
Number theory is a branch of pure mathematics devoted primarily to the study of the integers, sometimes called "The Queen of Mathematics" because of its foundational place in the discipline. Number theorists study prime numbers as well as the properties of objects made out of integers or defined as generalizations of the integers.
   
   1. is_prime(num) - Checks whether a given number num is a prime number or not.
   
   2. primes(limit) - Gives the list of all prime numbers less that the limit. Read http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
   
   3. Prime_Factorization(x) - Gives a list of all prime factors of the number x.
   
   4. fib(n) - Generates the n-th Fibonacci number in a fast way.
   
   5. LCM(a, b) - Counts the LCM for numbers a and b. For description read: http://en.wikipedia.org/wiki/Least_common_multiple
   
   6. Count_Totatives(n) - Counts the totatives a number n. Read: http://en.wikipedia.org/wiki/Totative
   
   7. GCD(a,b) - Counts the GCD for numbers a and b. For description read: http://en.wikipedia.org/wiki/Greatest_common_divisor
   
   8. Factor(n) - Provides all the factors of the number n.
   
   9. Pell_Equations(n, k) - Returns the list of first k solutions of the Pell's equation ( x^2 - n*(y^2) = 1 ).
   
   10. perfect(n) - Checks whether the number n is a perfect number or not. Read: http://en.wikipedia.org/wiki/Perfect_number
   

#### SEARCHING ALGORITHMS
In computer science, a search algorithm is an algorithm for finding an item with specified properties among a collection of items. The items may be stored individually as records in a database; or may be elements of a search space defined by a mathematical formula or procedure.
   
   1. Binary_Search(seq, key) - Finds the position of given input key in a sequence seq. Read: http://en.wikipedia.org/wiki/Binary_search_algorithm
   
   2. DFS(graph, start, path = []) - Implementation of DFS for traversing and searching the graph. Read: http://en.wikipedia.org/wiki/Depth-first_search
   
   3. KMP(string, word) - Searches the substring word in a string. Read: http://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm

   4. BMH(text, pattern) - Searches a substring pattern in a text. Read: http://en.wikipedia.org/wiki/Boyer%E2%80%93Moore%E2%80%93Horspool_algorithm

   5. RKS(substring, string) - Finds any one of a set of pattern strings in a text. Read: http://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm 

#### DATA ALGORITHMS

   1. shuffle(seq) - Shuffles all the items in the list.

   2. jaccard_index(dataset_1, dataset_2) - Calculates Jaccard similarity coefficient for two datasets. Read more: http://en.wikipedia.org/wiki/Jaccard_index

   3. Levenshtein_distance(s1, s2) - Measure the distance between two sequences. Read more: http://en.wikipedia.org/wiki/Levenshtein_distance

   4. find_max_subarray(numbers) - Finds the contiguous subarray within a one-dimensional array of numbers which has the largest sum.

   5. permutations(l) - A generator that retursn all the permutations of the given list l.

   6. is_polindrom(string) - Checks whether the given string is a polindrom or not.

   7. Most_Common(lst) - Identifies the most common element in the given list lst.

   8. is_Anagram(s1,s2) - Checks whether strings s1 and s2 are anagrams or not.

   9. Find_Nonrepeat(s) - Finds the first nonrepeated character in the string s.

   10. atoi(str) - Converts str into an integer, and returns that integer.

   11. Weighted_Random(W) - Generate random integers given a list W of integers.

   12. Hamming_Distance(str1, str2) - Calculates the dist
  

## References 
   
   1. http://rosettacode.org
   
   2. http://markmiyashita.com/cs61a/sp14

   3. http://www.laurentluce.com

   4. https://github.com/nryoung/algorithms

   5. http://interactivepython.org/runestone/static/pythonds/SortSearch/sorting.html
   
