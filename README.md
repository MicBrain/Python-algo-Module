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
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
