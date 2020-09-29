---
title: "What is a Hash?"
author: "Will Barker"
date: 2020-09-28
tags: ["comp-sci"]
---

Well, for starters, a hash is an important part of a very useful and fast data structure called a hash table. Let's reframe the question and ask what a hash table is instead.

<!--more--> 

## The idea behind a Hash Table

A hash table is a data structure that allows for the fast retrieval of data within it, regardless of how much is being stored. It's used to do many things, from efficient caching to database indexing and error checking. The central idea of a hash table can be explained with an example. Let's say we have a big array filled with random elements, one of which we want to retrieve:


```python
from numpy.random import randint


def create_array_with_hidden_string(array_size, hidden_string):
    random_numbers = randint(0, array_size**2, size=(array_size,))
    array = list(random_numbers)
    element_ix = randint(0, array_size)
    array.insert(element_ix, hidden_string)
    return array, element_ix

hidden_string = 'yeet'
array, hidden_string_ix = create_array_with_hidden_string(10**3, hidden_string)
```


```python
array[:10] # first 10 elements of the array
```




    [438536, 563467, 983737, 888887, 142932, 801221, 364091, 488761, 87409, 695441]



In the code above we've created an array with 1001 strings, one of which we want to find. Our string of interest, yeet, has been inserted at random index in our array. Let's find it. One way we could do so is through a linear search - simply iterating through the elements and stopping once we find yeet. Let's create a function that implements a linear search and reports how long it takes:


```python
import time

def linear_search_and_time(array, hidden_string):
    t_start = time.time()
    for i in range(len(array)):
        if array[i] == hidden_string:
            t_finish = time.time()
            t_delta = t_finish - t_start
            print(f"""found {array[i]} in array of size {len(array)} at index {i} in {round(t_delta,4)} seconds""")
```


```python
linear_search_and_time(array, hidden_string)
```

    found yeet in array of size 1001 at index 81 in 0.0004 seconds


Linear search certainly does the trick, but what if our array was of a much bigger size? A consideration of linear search is that the time it takes to complete scales, well, linearly with the size of the array being searched. In Big O Notation, the time complexity of a linear search is O(N):


```python
array_sizes = [10**4, 10**5, 10**6, 10**7]
for size in array_sizes:
    array, hidden_string_ix = create_array_with_hidden_string(size, hidden_string)
    linear_search_and_time(array, hidden_string)
```

    found yeet in array of size 10001 at index 2145 in 0.0085 seconds
    found yeet in array of size 100001 at index 28500 in 0.1093 seconds
    found yeet in array of size 1000001 at index 543310 in 2.1833 seconds
    found yeet in array of size 10000001 at index 9412368 in 36.1402 seconds


As we can see, the linear search gets slower as the size of the array grows.


Search is basically instantaneous if you know the position/index of the element you want to find in the array - no matter its size, an array lookup always takes a fast, constant time. In Big O terms, this is O(1):


```python
def index_lookup_and_time(array, hidden_string_ix):
    t_start = time.time()
    hidden_string = array[hidden_string_ix]
    t_finish = time.time()
    t_delta = t_finish - t_start
    print(f"""found {hidden_string} in array of size {len(array)} at index {hidden_string_ix} in {round(t_delta,4)} seconds""")
```


```python
for size in array_sizes:
    array, hidden_string_ix = create_array_with_hidden_string(size, hidden_string)
    index_lookup_and_time(array, hidden_string_ix)
```

    found yeet in array of size 10001 at index 4934 in 0.0 seconds
    found yeet in array of size 100001 at index 72609 in 0.0 seconds
    found yeet in array of size 1000001 at index 827164 in 0.0 seconds
    found yeet in array of size 10000001 at index 2401275 in 0.0 seconds


This idea of using the index of an element to access it is at the core of hash tables. Independent of the size of the hash table or the position of any of its elements, actions such as adding, accessing, or deleting have a constant O(1) time complexity.

## Hashing Algorithms

To achieve this, a unique index is created for each value in the table using a hashing algorithm. When a value is "hashed", it means that it is being passed through an algorithm that transforms it into a memory address (i.e. an index position). This address should never change over the lifetime of the element and is calculated using aspects of both the element itself and the hash table it is being placed into. In Python's implementation of object hashing, this requirement means that only immutable (i.e. unchangeable) data types like strings, ints, and tuples can be hashed. If mutable objects were able to be hashed, then the hashing algorithm would generate different indexes for them as they changed and internal consistency would be lost.

When an element is added, deleted, or retrieved from a hash table it is first passed into a hashing algorithm to find its location. Whatever action needs to happen then takes place after that. 

There are different ways to implement hashing algorithms, but in principle they should all be easy to calculate as to keep things fast. Algorithms can also differ based on the intended inputs' data types. For numeric keys, a simple hashing function might divide the key by the number of available addresses within an array (i.e. its size) and take the remainder:


```python
# first, let's create an empty array of a certain size

def make_empty_hash_table(size: int):
    return [None] * size
```


```python
array = make_empty_hash_table(5)
print(array)
```

    [None, None, None, None, None]



```python
# create our number hashing algorithm

def simple_numeric_hash(num: int, array_size: int):
    return num % array_size
```


```python
# check indices generated by elements

print(simple_numeric_hash(30, len(array)))
print(simple_numeric_hash(21, len(array)))
print(simple_numeric_hash(9, len(array)))
```

    0
    1
    4



```python
# define function to insert the elements at the indices provided by the hashing algorithm

def insert_number_into_hash_table(num, array):
    ix = simple_numeric_hash(num, len(array))
    array[ix] = num
    return array
```


```python
array = insert_number_into_hash_table(30, array)
array = insert_number_into_hash_table(21, array)
array = insert_number_into_hash_table(9, array)

print(array)
```

    [30, 21, None, None, 9]


For a string key, it might sum the ASCII codes for each character in the string and do the same division by the array size, taking the remainder:


```python
# create another empty array

array = make_empty_hash_table(6)
```


```python
# create our string hashing algorithm

def simple_string_hash(string: str, array_size: int):
    ascii_sum = 0
    for char in string:
        ascii_sum += ord(char)
    return ascii_sum % array_size
```


```python
# check indices generated by pet hamster names

print(simple_string_hash('Squeeky', len(array)))
print(simple_string_hash('Squishy', len(array)))
print(simple_string_hash('Squirrelly', len(array)))
```

    5
    2
    4



```python
# define function to insert the elements at the indices provided by the hashing algorithm

def insert_string_into_hash_table(string, array):
    ix = simple_string_hash(string, len(array))
    array[ix] = string
    return array
```


```python
# store some pet hamsters in a hash table

array = insert_string_into_hash_table('Squeeky', array)
array = insert_string_into_hash_table('Squishy', array)
array = insert_string_into_hash_table('Squirrelly', array)

print(array)
```

    [None, None, 'Squishy', None, 'Squirrelly', 'Squeeky']


## Managing Collisions

A hashing algorithm is an elegant way to sort data and have it be quickly accessible. There is one problem that needs to be solved with this approach, however - what if two elements generate the same hash?


```python
print(simple_string_hash('Squishy', len(array)))
print(simple_string_hash('Smokey', len(array)))
```

    2
    2


This is what's known as a collision - when multiple elements yield the same hash. Positions in a hash table can't belong (at least directly) to more than a single element, so available positions are used up as elements are added in. Fortunately, there are a variety of solutions that hash tables can use to manage collisions. These solutions fall into two categories: open addressing and closed addressing.


Open addressing solutions will place a collided element in another open position within the hash table. One implementation of this is linear probing, which will do a linear search from the collided index for the next available spot and place the element there. When the element needs to be accessed, the hash table will compute its hash, find the index position, see that position is occupied by something else, and then do a linear search from that spot to find it. Other open addressing algorithms behave similarly, just switching up how the search is performed: plus 3 rehashing checks every third available position, quadratic probing squares the number of failed placement attempts and checks that many positions away, and double hashing applies a second hash function after the first yields a collision. With all of these approaches the goal is find an available position further away from the collision - uniformly distributing across the table will reduce the likelihood of further collisions happening.

Here's a simple implementation of open addressing with linear probing:


```python
def linear_probing_insert(element, ix, array):
    if array[ix] is None:
        array[ix] = element
    else:
        print(f'collision at index {ix}')
        lin_search_positions = list(range(ix + 1, len(array))) + list(range(0, ix))
        for pos in lin_search_positions:
            if array[pos] is None:
                array[pos] = element
                print(f'added {element} at index {pos} instead')
                return array
        print(f'nowhere to put {element}')
        return array

def insert_with_open_addressing(string, array):
    ix = simple_string_hash(string, len(array))
    return linear_probing_insert(string, ix, array)
```


```python
open_example = array
```


```python
insert_with_open_addressing('Smokey', open_example)
```

    collision at index 2
    added Smokey at index 3 instead





    [None, None, 'Squishy', 'Smokey', 'Squirrelly', 'Squeeky']



Closed addressing solutions create linked lists at occupied positions in the hash table and add elements sequentially within them. Whereas open addressing solutions aim to place elements across the entire hash table's available positions, closed addressing solutions opt to nest multiple elements within a sub-level list at the positions themselves (think going deep instead of going wide). Either way, both approaches end up having to do some sort of search after hashing an element and getting a collided position.

Here's a simple implementation of closed addressing with linked lists:


```python
def closed_addressing_insert(element, ix, array):
    if array[ix] is None:
        array[ix] = [element]
    else:
        print(f'collision at index {ix}')
        array[ix].append(element)
        print(f'added {element} in a nested list at index {ix}')

def insert_with_closed_addressing(string, array):
    ix = simple_string_hash(string, len(array))
    return closed_addressing_insert(string, ix, array)
```


```python
closed_example = make_empty_hash_table(6)
```


```python
insert_with_closed_addressing('Squishy', closed_example)
insert_with_closed_addressing('Squirrelly', closed_example)
insert_with_closed_addressing('Squeeky', closed_example)
insert_with_closed_addressing('Smokey', closed_example)
```

    collision at index 2
    added Smokey in a nested list at index 2



```python
closed_example
```




    [None, None, ['Squishy', 'Smokey'], None, ['Squirrelly'], ['Squeeky']]



One can aim to avoid collisions entirely by using a larger hash table with more available positions than values to occupy them. The ratio between stored elements and available positions in a hash table is called its load factor, and the lower this is the lower the likelihood of collisions (and subsequent linear searches). If a hash table contains too many collisions, its performance will worsen as more searches with increased time complexity are needed. No collisions means that access time will always be a speedy O(1) time complexity. A good hashing algorithm should aim to minimize collisions, resolve them quickly if they do happen, and generate a uniform distribution of hash values across the table.

## The Difference between a Hash Table and a Hash Map

A map is an association between a key and a value - for key K remember value V.

Mapping between keys and values can be done in many different ways, and one way of doing so is with a hash table. A hash table is just a structure for storing data, but when it stores data that is in the form of a distinct key-value pairing, it's called a hash map.

In a hash table that simply stores values like the one above, there are no key-value pairs; each value is its own key with no association with another object.

In essence, a hash table is a data structure for quickly storing and accessing a data. A hash map is a particular kind of hash table that stores key-value pairs.

## Hash Map Implementation with Lists

There are three components to this Hash Map:
- Hash Table: a simple array
- Hashing Algorithm: a simple hash function that sums the ASCII values in a string, divides it by the size of the array, and returns the remainder
- Collision Handling: A closed addressing solution for adding elements into a list at each position, and then performing a linear search at the time of accessing


```python
class HashMap:
    def __init__(self, size):
        self.size = size
        self.map = [None] * self.size
    
    def _get_hash(self, key):
        hash_ = 0
        for char in str(key):
            hash_ += ord(char)
        return hash_ % self.size
    
    def add(self, key, value):
        key_hash = self._get_hash(key)
        key_value = [key, value]
        
        if self.map[key_hash] is None:
            self.map[key_hash] = list([key_value])
            return True
        else:
            for pair in self.map[key_hash]:
                if pair[0] == key:
                    pair[1] = value
                    return True
            self.map[key_hash].append(key_value)
            return True
        
    def get(self, key):
        key_hash = self._get_hash(key)
        
        if self.map[key_hash] is not None:
            for pair in self.map[key_hash]:
                if pair[0] == key:
                    return pair[1]
        return None
    
    def delete(self, key):
        key_hash = self._get_hash(key)
        
        if self.map[key_hash] is None:
            return False
        
        for i in range(0, len(self.map[key_hash])):
            if self.map[key_hash][i][0] == key:
                self.map[key_hash].pop(i)
                return True
    
    def display(self):
        for item in self.map:
            if item is not None:
                print(str(item))
```


```python
# lists of hamsters and their favourite foods - the key value pairs to be inserted into the hash map

hamsters = ['Squeeky', 'Squishy', 'Speedy', 'Smokey', 'Squirrelly',
            'Snowy', 'Sleepy', 'Smoochy', 'Smarty', 'Snoozy', 'Smoothy', 'Shiny']

fav_foods = ['Pizza', 'Corn', 'Baguettes', 'Beef Jerky', 'Pho', 'Pad Thai',
             'Shawarma', 'Chocolate', 'Carrots', 'Wine', 'Smoothies', 'Apples']
```

To demonstrate the idea of a hash table's load factor, we'll create two hash maps - one with a size equalling the number of elements, and one with a size that is the square of the number of elements.


```python
hamster_hash_map = HashMap(len(hamsters))

for hamster, food in zip(hamsters, fav_foods):
    hamster_hash_map.add(hamster, food)
```


```python
hamster_hash_map.display()
```

    [['Squishy', 'Corn'], ['Sleepy', 'Shawarma']]
    [['Squirrelly', 'Pho'], ['Snowy', 'Pad Thai'], ['Smarty', 'Carrots']]
    [['Speedy', 'Baguettes'], ['Smoochy', 'Chocolate']]
    [['Shiny', 'Apples']]
    [['Smokey', 'Beef Jerky']]
    [['Snoozy', 'Wine']]
    [['Squeeky', 'Pizza'], ['Smoothy', 'Smoothies']]


When the hash table is small relative to the number of elements being stored, collisions can occur. Not a big deal with only a few elements, but at much larger scales this would begin to hurt performance.


```python
big_hamster_hash_map = HashMap(len(hamsters)**2)

for hamster, food in zip(hamsters, fav_foods):
    big_hamster_hash_map.add(hamster, food)
```


```python
big_hamster_hash_map.display()
```

    [['Smoochy', 'Chocolate']]
    [['Squeeky', 'Pizza']]
    [['Smoothy', 'Smoothies']]
    [['Squishy', 'Corn']]
    [['Speedy', 'Baguettes']]
    [['Sleepy', 'Shawarma']]
    [['Smokey', 'Beef Jerky']]
    [['Smarty', 'Carrots']]
    [['Squirrelly', 'Pho']]
    [['Snoozy', 'Wine']]
    [['Shiny', 'Apples']]
    [['Snowy', 'Pad Thai']]


When the size of the hash table is much larger than the number of elements, there's a reduced chance of collisions occuring. This is an ideal scenario where the table is highly performant an increased likelihood of O(1) access times.
