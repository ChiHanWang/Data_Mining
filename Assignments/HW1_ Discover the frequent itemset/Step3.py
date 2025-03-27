import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import time
from optparse import OptionParser
import os

def writeOutputToFile(inFile: str, outFile: str, minSupport: float, patterns: pd.DataFrame):
    dataset_name = os.path.basename(inFile).split('.')[0]
    output_filename1 = os.path.join(outFile, f"step3_task1_{dataset_name}_{minSupport}_result1.txt")
    output_filename2 = os.path.join(outFile, f"step3_task1_{dataset_name}_{minSupport}_result2.txt")
    
    sorted_patterns = patterns.sort_values(by='support', ascending=False)
    
    with open(output_filename1, 'w') as f:
        for _, row in sorted_patterns.iterrows():
            item_str = "{%s}" % ",".join(map(str, row['itemsets']))
            f.write("%.1f\t%s\n" % (row['support'] * 100, item_str))
    
    with open(output_filename2, 'w') as f:
        f.write("%d" % len(patterns))

if __name__ == "__main__":

    start_time= time.time()
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
        "-p", "--outputPath", dest="output", help="output file path", default='./'
    )

    (options, args) = optparser.parse_args()

    try:
        os.makedirs(options.output)
    except:
        pass
    
    data_path = options.input
    transactions = []
    with open (data_path, 'r') as data:
        for lines in data:
            str_line = list(lines.strip().split())
            transactions.append(str_line[3:])
    # print("--- Elapsed time of load data : %s seconds ---" %(time.time() - start_time))
    
    # transform format
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    # print("--- Elapsed time of data transform : %s seconds ---" %(time.time() - start_time))

    min_support = options.minS

    # fpgrowth
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    # print("--- Elapsed time of fpgrowth : %s seconds ---" %(time.time() - start_time))

    writeOutputToFile(data_path, options.output, options.minS, frequent_itemsets)

    print("Results of task1 of Step3 :")
    print("--- Elapsed time of task1 : %s seconds ---" %(time.time() - start_time))
