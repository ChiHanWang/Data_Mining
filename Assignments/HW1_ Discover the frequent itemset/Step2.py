"""
Description     : Simple Python implementation of the Apriori Algorithm
Modified from:  https://github.com/asaini/Apriori
Usage:
    $python Step2.py -f DATASET.data -s minSupport

    $python Step2.py -f DATASET.data -s 0.01
"""

import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
import os
import time

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """Calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)
     
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1
    
    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet


def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )


def getItemSetTransactionList(data_iterator):
    """Get itemset of each transaction list"""
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    
    return itemSet, transactionList


def runApriori(data_iter, minSupport):
    """
    Run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - num_iter {iter, [num of candidates before pruning, num of candidates after pruning]}
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    num_iter = defaultdict(list)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport
    num_iter[1].append(len(itemSet))
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)
    num_iter[1].append(len(oneCSet))
    
    currentLSet = oneCSet
    k = 2
    while currentLSet != set([]):    
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        num_iter[k].append(len(currentLSet))
        currentCSet = returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet
        )
        num_iter[k].append(len(currentCSet))
        currentLSet = currentCSet
        k = k + 1
    

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    return toRetItems, num_iter

def find_closed_frequent_itemsets(frequent_itemsets):
    """
    Run the finding closed frequent itemsets algorithm
    Retuen:
    -closed_itemsets(itemset, support)
    """
    closed_itemsets = []

    for itemset, support in frequent_itemsets:
        is_closed = True
        for other_itemset, other_support in frequent_itemsets:
            if set(itemset) < set(other_itemset) and support == other_support:
                is_closed = False
                break

        if is_closed:
            closed_itemsets.extend([(itemset, support)])

    return closed_itemsets

def writeOutputToFile_task1(inFile, outFile, minSupport, items, num_iter):
    """Writes and stores task1 output in .txt"""
    dataset_name = os.path.basename(inFile).split('.')[0]
    output_filename = os.path.join(outFile, f"step2_task1_{dataset_name}_{minSupport}_result1.txt")
    with open(output_filename, 'w') as f:
        for item, support in sorted(items, key=lambda x: x[1], reverse=True):
            item_str = "{%s}" % ",".join(map(str, item))
            f.write("%.1f\t%s\n" %(support*100, item_str))

    output_filename = os.path.join(outFile, f"step2_task1_{dataset_name}_{minSupport}_result2.txt")
    with open(output_filename, 'w') as f:
        f.write("%s\n" %(len(items)))
        for key, value in num_iter.items():
            f.write("%s\t%s\t%s\n" %(key, value[0], value[1]))

def writeOutputToFile_task2(inFile, outFile, minSupport, closed_itemset):
    """Writes and stores task2 output in .txt"""
    dataset_name = os.path.basename(inFile).split('.')[0]
    output_filename = os.path.join(outFile, f"step2_task2_{dataset_name}_{minSupport}_result1.txt")
    with open(output_filename, 'w') as f:
        f.write("%s\n" %(len(closed_itemset)))
        for item, support in sorted(closed_itemset, key=lambda x: x[1], reverse=True):
            item_str = "{%s}" % ",".join(map(str, item))
            f.write("%.1f\t%s\n" %(support*100, item_str))

def printResults(items):
    """prints the generated itemsets sorted by support """
    for item, support in sorted(items, key=lambda x: x[1]):
        print("item: %s , %.3f" % (str(item), support))

def to_str_results(items):
    """prints the generated itemsets sorted by support"""
    i = []
    for item, support in sorted(items, key=lambda x: x[1]):
        x = "item: %s , %.1f" % (str(item), support)
        i.append(x)
    return i


def dataFromFile(fname):
    """Function which reads from the file and yields a generator"""
    with open(fname, "r") as file_iter:
        for line in file_iter:
            # line = line.strip().rstrip(",")  # Remove trailing comma
            record = frozenset(line.split()[3:])
            yield record


if __name__ == "__main__":
    
    start_time = time.time()    
    optparser = OptionParser()
    # adding inputFile option for command line
    optparser.add_option(
        "-f", 
        "--inputFile", 
        dest="input", 
        help="filename containing .data", 
        default='datasetA.data'
    )
    # adding minSupport option for command line
    optparser.add_option(
        "-s",
        "--minSupport",
        dest="minS",
        help="minimum support value",
        default=0.01,
        type="float",
    )
    # adding outputPath option for command line
    optparser.add_option(
        "-p", 
        "--outputPath", 
        dest="output", 
        help="output file path", 
        default='./'
    )
    
    (options, args) = optparser.parse_args()

    try:
        """ 根據使用者的命名來 create a folder"""
        os.makedirs(options.output)
    except:
        pass

    inFile = None
    if options.input is None:
        inFile = sys.stdin
    elif options.input is not None:
        inFile = dataFromFile(options.input)
    else:
        print("No dataset filename specified, system with exit\n")
        sys.exit("System will exit")

    minSupport = options.minS

    items, num_iter = runApriori(inFile, minSupport)

    writeOutputToFile_task1(options.input, options.output, minSupport, items, num_iter)
    print("Results of task1 of Step2 :")
    print("--- Elapsed time of task1 : %s seconds ---" %(time.time() - start_time))

    closed_itemset = find_closed_frequent_itemsets(items)

    writeOutputToFile_task2(options.input, options.output, minSupport, closed_itemset)

    print("Results of task2 of Step2 :")
    print("--- Elapsed time of task2 : %s seconds ---" %(time.time() - start_time))
