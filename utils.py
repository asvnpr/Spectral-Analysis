import numpy as np
import pandas as pd 
from scipy import special
import scipy.sparse
import types
from itertools import combinations
import pylab as plt
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import csv
import os
import sys
from matplotlib.legend import Legend
from matplotlib.pyplot import figure
from operator import itemgetter, attrgetter
import random
import string
import math

# input: string, string
# output: matrix
# Uploads dataset from file path (first string) and reduces it using the method specified in the 
# second string, returns reduced dataset. Does not alter original data.
def reduce_data(filePath, reduction):
    # upload data and prepare reduced dataset
    geneDat = pd.read_csv(filePath, index_col='Unnamed: 0', encoding='utf-8')
    matrix = geneDat.values
    geneMut = np.zeros((matrix.shape[0],int(np.ceil((matrix.shape[1])/2))))
    
    # will go through each row
    for rows in range(matrix.shape[0]):
        # go through each column until the second to last column (excludes phenotype)
        for cols in range(0,matrix.shape[1]-1,2):
            # checks to see if either the first or second copy is present (represented as a 1)
            # reduce dataset
            # reason for int(cols/2) is to match upindexes with the pairing number that it is on
            # ex: col 0 and 1 output value will be represented in the col 0 of reduced dataset. 
            # And col 2 and 3 gene output value will be represented in the col 1 of reduced dataset
            if(reduction == "Heterozygous"):
                if(matrix[rows][cols] == 1 ^ matrix[rows][cols+1] == 1):
                    geneMut[rows][int(cols/2)] = 1
            elif(reduction == "Homozygous"):
                if(matrix[rows][cols] == 1 and matrix[rows][cols+1] == 1):
                    geneMut[rows][int(cols/2)] = 1
            elif(reduction == "Mutation Presence"):
                if(matrix[rows][cols] == 1 or matrix[rows][cols+1] == 1):
                    geneMut[rows][int(cols/2)] = 1
            else:
                print("Choose a valid option for reduction of the dataset.")

    # assign last column from original matrix as last column for new matrix (new dataset)
    geneMut[:,-1] = matrix[:,-1]
    return geneMut

# input: matrix
# output: list
# Takes in a matrix and returns a list that contains the numbers 
# of zeros on each line of the matrix correspondingly
def num_of_mutations(RDM):
    # make copy of data to not alter 
    tempRDM = np.copy(RDM)
    tempRDM = np.delete(tempRDM,-1,1)
    # list to count the number of ones
    nones = []
    nrows = RDM.shape[0]
    # for each row, sum the rows and save in list of ones
    for row in range(nrows):
        nones.append(sum(tempRDM[row,]))
    # create list of zeros by subracting the total amount of 
    # columns by the ones
    nzeros = [tempRDM.shape[1]]*(len(nones))
    nzeros = [a_i - b_i for a_i, b_i in zip(nzeros, nones)]
    # return list of ceros
    return nzeros

# input: matrix, list
# output: list of matrices
# Takes in a matrix and divides it into smalles matrices that 
# have the same amount of zeron in each line
def raw_data_partitions(RDM, nzeros):
    nrows = RDM.shape[0]
    ncols = RDM.shape[1]
    # variable to store matrices
    fPartitions = []
    # get unique values of the amount of zeros 
    distinctAmountsZeros = np.unique(nzeros)
    distinctAmountsZeros = np.sort(distinctAmountsZeros)
    # for each amount of zeros
    for amount in distinctAmountsZeros:
        listAmounti = []
        # check every row if compatible with amount of zeros
        for row in range(nrows):
            if(nzeros[row]==amount):
                # and store them
                listAmounti.append(RDM[row,])
        # to concatenate into a matrix
        MatrixAmounti = np.concatenate(listAmounti).reshape(len(listAmounti), ncols)
        # and append it to the list of smaller matrices
        fPartitions.append((amount, MatrixAmounti))
    # return list of smaller matrices
    return fPartitions

# input: list of tuples of float and matrix
# output: two lists of tuples of float and matrix
# Takes in a list of tuples that contain k 
# and data matrix with k zeros in each row and 
# separates them into two lists one with majority 
# zeros the other with minority zeros
def minorities_majorities(dataPartition):
    Minorities = []
    Majorieties = []
    nvars = dataPartition[0][1].shape[1] - 1
    limit = np.floor(nvars/2)
    for elem in range(len(dataPartition)):
        k = dataPartition[elem][0]
        if (k > limit):
            Majorieties.append(dataPartition[elem])
        else:
            Minorities.append(dataPartition[elem])
    return Minorities, Majorieties

# input: list of tuples of length two (float, matrix)
# output: list of tuples of length two (float, matrix)
# This function takes in a list of tuples where the 
# first value is the amount of zeros on each row (denoted k) of the 
# matrix (second value of tuple) and it flips the matricies
# i.e. changes all the 1's to 0's and vice versa
# it also changes the k to the total amount of variables 
# (denoted n) minus k (so it does keep count of the amount of zeros)
def flip_majorities(MajorityData):
    NewMinorities = []
    for tupIndex in range(len(MajorityData)):
        k = MajorityData[tupIndex][0]
        data = MajorityData[tupIndex][1]
        ncols = data.shape[1]-1
        nrows = data.shape[0]
        newMatrix = np.zeros((nrows, ncols+1))
        for i in range(nrows):
            for j in range(ncols):
                if(int(data[i,j]) == 1):
                    newMatrix[i,j] = 0
                elif(int(data[i,j]) == 0):
                    newMatrix[i,j] = 1
        # copy hemoglobin as is
        for h in range(nrows):
            newMatrix[h, -1] = data[h, -1]
        NewMinorities.append( (ncols-k, newMatrix) )
    return NewMinorities

# input: one dimensional matrix (horizontal)
# output: tuple or int
# Takes in a row and categorizes the row by the amount of zeros 
# and returns the corresponding variables to the zeros 
# e.g. the row [1,0,1,0] with the columns 1234, return (2,4)
# e.g. the row [0,1,1,1] with columns 1234 returns 1
def categorize_row(row):
    # copy row to not alter the data
    tempRow = np.copy(row)
    tempRow = np.asarray(tempRow)
    # length of row without including the last column (which are the results)
    length = len(tempRow)-1 # minus hemoglobin
    # variable to store the variables corresponding to the zeros
    category = []
    # check line for zeros
    for i in range(length):
        if(int(tempRow[i]) == 0):
            category.append(i)
    # if it is only one zero in the row it returns the int
    if(len(category) != 1):
        # returns tuple of category
        return tuple(category)
    else:
        return category[0]
    
# input: tuple of int and matrix
# output: array
# Takes in a tuple of the number of zeros in the matrix and 
# the matrix and calculates f^(n, n-#0s)
def calc_f(PartitionTuple):
    # save all the variables
    k = PartitionTuple[0]
    data = PartitionTuple[1]
    nrows = data.shape[0]
    nvars = data.shape[1]-1
    # list to append the categories to
    rowCategory = []
    # for individual values the row categories are calculated
    # differently because we dont have to use the function 
    # combinations for the lexicographical order
    if(k == 1):
        # counter to know how many hemoglobin we add so we can average them
        counter = np.zeros(int(nvars))
        f = np.zeros(int(nvars))
        lexOrder = list(range(int(nvars)))
    else:
        # counter to know how many hemoglobin we add so we can average them
        counter = np.zeros(int(special.binom(nvars,nvars-int(k))))
        f = np.zeros(int(special.binom(nvars,nvars-int(k))))
        lexOrder = list(combinations(list(range(nvars)),int(k)))
    # for every row in the matrix add a category for each line
    # to rowCategory list
    for row in range(nrows):
        cat = categorize_row(data[row,])
        rowCategory.append(cat)
    # for every row in the data
    for row in range(nrows):
        # go through all the orgers
        for order in range(len(lexOrder)):
            # and check if the categories match
            if(rowCategory[row] == lexOrder[order]):
                # add one to the counter
                counter[order] = counter[order] + 1
                # if they do, add the results (phenotype) to the data vector
                f[order] = f[order] + data[row,-1]    
    # divide data vector by the counter to average it
    averagef = np.zeros(len(f))
    # be sure not to divide by zero
    for i in range(len(averagef)):
        if(int(counter[i]) == 0):
            averagef[i] = f[i]
        else:
            averagef[i] = f[i]/counter[i]
    # return data vector
    return averagef, counter

# input: tuple of int and matrix
# output: list of arrays
# Caluclates f^(n, n-k) for all ks and returns them 
# in a list of data vectors
def create_data_vectors(PartitionList):
    dataVectors = []
    counterVectors = []
    for data in range(len(PartitionList)):
        datavec, counter = calc_f(PartitionList[data])
        dataVectors.append(datavec)
        counterVectors.append(counter)
    return dataVectors, counterVectors

# input: two list of vectors
# output: average
# takes the average of the noncero entries of all the vectors in both list of vectors
def calc_average(minDv, majDv):
    # counters
    counter = 0
    sumDv = 0
    # go through each data vector and add to counters
    for i in range(len(minDv)):
        sumDv += sum(minDv[i])
        # only count entries distinct from cero
        for j in range(len(minDv[i])):
            if(int(minDv[i][j]) != 0):
                counter +=1
    # repeat for data vectors of majority data
    for i in range(len(majDv)):
        sumDv += sum(majDv[i])
        for j in range(len(majDv[i])):
            if(int(majDv[i][j]) != 0):
                counter +=1
    # compute average
    average = sumDv/counter
    return average

# input: list of vectors, int
# output: list of vectors
# creates a new list of vectors where the entries are the same except the zero 
# entries which will be imputed with the average (int)
def impute_data_vector(dataVectors, avg):
    # new list of data vectors
    imputedVectors = []
    # for every data vector
    for i in range(len(dataVectors)):
        # create zero data vector
        imputed = np.zeros(len(dataVectors[i]))
        for j in range(len(dataVectors[i])):
            # if entry is zero, impute
            if(int(dataVectors[i][j]) == 0):
                imputed[j] = avg
            # else save the same entry
            else:
                imputed[j] = dataVectors[i][j]
        # apend imputed vector to new data vector list
        imputedVectors.append(imputed)
    # return imputed data vectors list
    return imputedVectors

# input: string or list, int
# output: list of tuples
# This function returns all the combination of the characters in the string 
# of length k as a list of tuples. It can also take a list as its first 
# argument and has the same output but treats the elements of the list 
# like characters in a string. It should be noted that this function
# "renames" the function combinations of the itertools library
# so it is more readable to the user.
def alphabet_choose_k(alphabet, k):
    return list(combinations(alphabet, int(k)))

# input: list of same length tuples
# output: matrix
# Takes in a tuple list each of the same size k which are all the 
# combinations of our variables and calculates the adjacency matrix
# of its Johnson Graph 
def calc_adj_mat(tupleList):
    # type check the tuple list
    if isinstance(tupleList, list)==False:
        return("Argument is not a list!")
    # Create adjacency matrix of size of the amount of tuples in the 
    # list on both columns and rows
    dim = len(tupleList)  
    # SparseAdj = dok_matrix((dim,dim))
    Adj = np.zeros((dim,dim))
    # Go through the matrix 
    for i in range(dim):
        for j in range(dim):
            # if the intersection of the nodes (which are the tuples) 
            # is equal to their length - 1 then they are connected
            # and we put a one in the matrix to represent that connection
            if len(set(tupleList[i]).intersection(set(tupleList[j]))) == len(tupleList[0])-1:
                # SparseAdj[i,j] = 1
                Adj[i,j] = 1
    return Adj

# input: matrix
# output: list of floats, list of arrays
# Takes in an adjacency matrix (which is symmetric) and calculates
# the eigen values and eigen vectors and returns them as a list of floats
# and as a list of arrays respectively. It should be noted that the 
# eigen values corrspond with the same indexed eigen vectors.
# This function "renames" the function np.linalg.eigh for the
# purpose of making it easier for the user.
def calc_eig_vec_of_adj_matrix(mat):
    eigVal, eigVec = np.linalg.eigh(mat)
    return eigVal, eigVec

# input: list of floats
# output: list of arrays
# Since the eigen vectors corresponding to the same M_i in a 
# given decomposition (n,k) have the same eigen value, this
# function (that takes in eigen vectors and eigen values in 
# order of correspondance) concatenates (as matrices) eigen 
# vectors that correspond to the same M_i and returns these 
# as a list of matricies that are the basis for the M_i's
# in ascending order (i.e. [M_0, M_1,...,M_k])
def group_eig_vecs(eVals, eVecs):
    # gather the distinct eigen values and sort them
    distinctVals = np.unique(eVals.round(decimals=4))
    distinctVals = np.sort(distinctVals)
    # create empty list where we will store the basis
    MBasis = []
    # for every distinct eigen value we will go through the 
    # list of eigen values and if they are equal we will add the
    # corresponding eigen vector to a list 
    for val in distinctVals:
        # list for storing eigen vectors that correspond to 
        # the same M_i
        vecList = []
        for i in range(len(eVals)):
            if(val == eVals[i].round(decimals=4)):
                vecList.append(np.transpose(eVecs)[i])
        # transform list of arrays into matrix to append to M basis list
        vecList = np.array(vecList).reshape(len(vecList), len(vecList[0]))
        MBasis.append(vecList)
    # return M basis
    MBasis = np.flipud(MBasis)
    return MBasis

# input: array, list of matrices
# output: list of arrays
# This functions calculates the projection of the data vector 
# f onto each of the M_i basis (in the form of a matrix)
def proj_f_onto_M(f, MBasis):
    # emptylist to store the f_is as we calculate them 
    f_is = []
    # for every element of the list of M basis we will 
    # calculate f_i and append it to the list
    for i in range(len(MBasis)):
        f_is.append(proj_onto_M_i(f, MBasis[i]))
    # return list of f_i's
    return f_is

# input: array, matrix
# output: array
# This function takes in a data vector and a basis for M_i and 
# projects f onto that basis to calculate f_i
def proj_onto_M_i(f, M_i):
    # get the dimensions of M_i
    nrow = M_i.shape[0]
    ncol = M_i.shape[1]
    # create a zero vector to add the projections calculated by
    # components in the eigen vectors 
    f_i = np.zeros(ncol)
    # for each row of the basis (recall we saved the eigen vectors
    # as lines) we calculare the projections
    for j in range(nrow):
        # temporarily reshape the row to transform it from an array
        # to a one dimentional matrix (vertical)
        tmp = np.copy(M_i[j])
        tmp = tmp.reshape(len(M_i[j]),1)
        # add the new projection to f_i
        f_i = f_i + project(f, tmp)
    # return f_i
    return f_i
    
# input: one dimentional matrix, array
# output: one dimentional matrix
# This funtion calculates the projection of f onto b
def project(f, v):
    # reshape v to be a one dimentional matrix compatible 
    # for dot (or inner) product (horizontal)
    v = v.reshape(1,len(v))
    # calculate projection and return it
    result = (np.dot(v,f)/np.dot(v,np.transpose(v)))*v
    return result

# input: list (of ints or strings) or string, list, tuple
# output: list of arrays
# Takes in a list of data vectos given a partition (n,k)
# k being the amount of zeros in a row, implements Mallow's
# method and returns a list of data vectors as arrays
def Mallows_list(alphabet, f_iList, partition):
    # partition = (n,k)
    k = partition[1]
    # first data vector f_0 does nor change 
    # so we append it intact
    MallowsList = [np.concatenate(f_iList[0])]
    # for every data vecto f_i, 0<i<k-1, implement 
    # Mallows method
    for f_iIndex in range(1,k):
        MallowsList.append(calc_Mallow(alphabet, f_iList[f_iIndex], f_iIndex, k))
    # last data vecto f_k does not change so we append it
    # without changes
    MallowsList.append(np.concatenate(f_iList[-1]))
    # return mallows list
    return MallowsList

# input: list (of ints or strings) or string, array, int
# output: array
# Takes in a data vector f_i and returns Mallow's f_i
def calc_Mallow(alphabet, f_i, i, k):
    # variables
    rows = list(combinations(alphabet, i))
    cols = list(combinations(alphabet, k))
    # zero array for storing values
    Mallowf_i = np.zeros(len(rows))
    # T matrix just sums the values of the data vector 
    # where the values of (n choose i) (rows) is a subset
    # of the partition (n choose k) (columns)
    for iElem in range(len(rows)):
        for kElem in range(len(cols)):
            if(set(rows[iElem]).issubset(set(cols[kElem]))):
                # add elements of the data vector to 
                # Mallows data vector
                Mallowf_i[iElem] = Mallowf_i[iElem] + f_i[0,kElem]
    # return Maloow's data vector
    return Mallowf_i

# input: raw copied data as a matrix
# output: a list of summation for each observation and the number of mutations
def num_gene_mutations(raw_data):
    n_ones=[]
    # 0 to number of total rows
    for row in range(0,(raw_data.shape[0])):
        #calculating each person's number of gene mutations, excluding last column with phenotype levels
        n_ones.append(sum(raw_data[row,:-1]))
    return n_ones

# input: bool, string
# output: void
# prints a histogram and if bool = 1 saves it in the file named like the string
# If no file exists, it will give an error
def histogram_mutations(gene_mut, reduction, save, file):
    # Will need "num_gene_mutations" function in order to develop histogram 
    # of number of people for each number of mutations
    # gene_mut = string.ascii_uppercase[:num_muts]
    sum_gene_mut = num_gene_mutations(gene_mut)
    plt.hist(sum_gene_mut, bins = 15, rwidth = 1)
    # Labeling histogram
    if(reduction == "Heterozygous"):
        plt.title('Histogram of Total Gene Mutations \n Heterozygous Reduction')
    elif(reduction == "Homozygous"):
        plt.title('Histogram of Total Gene Mutations \n Homozygous Reduction')
    elif(reduction == "Mutation Presence"):
        plt.title('Histogram of Total Gene Mutations \n Reduction by Mutation Presence')
    plt.xlabel('Number of Gene Mutations')
    plt.ylabel('Number of People')
    if(bool(save) == 1):
        plt.savefig("%s/Histogram_%s.png"%(file,reduction))
    plt.show()

# input: int
# output: list of strings
# takes an int n and calculated n distinct colors in hexadecimal 
# and returns them as a list of strings where each string is a color 
# in hexadecimal. This function was taken from 
# https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
def colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        c = '#%02x%02x%02x' %(r,g,b)
        ret.append(c) 
    return ret

# input: string, list of matrices, bool, string
# output: void
# first string is the phenotype responce (e.g. Hemoglobin) with capital letter prefereably
# second argument is the partitioning of the original data by number of mutations per subject
# third argument is the option to save the scatterplot made
# fourth argument is the file name where the scatterplot will be saved
# if third argument is 0, it does not matter what string is given as a fourth argument
def scatterplot_mutations(pheno, dataList, save, file):
    # empty list for all phenotype levels of each partition in dataList
    phenotype_levels = []
    # get the number of mutations
    num_muts = dataList[0][1].shape[1]-1
    # for each partition we create a list of the phenotype levels
    for partition in range(len(dataList)):
        phenotype_response = []
        for subject in range(len(dataList[partition][1])):
            phenotype_response.append(dataList[partition][1][subject][-1])
        # we append each list as a tuple where the first element is the amount of 
        # mutations per row in that partition, and the second is the list of phenotype levels
        phenotype_levels.append( (num_muts-dataList[partition][0], phenotype_response) )
    # we call colors function to create n random colors where n is the amount of partitions in dataList
    colrs = colors(len(dataList))
    for grouping in range(len(dataList)):
        phenotype = phenotype_levels[grouping]
        # we color each partition differently in the scatterplot
        plt.scatter(x=range(len(phenotype[1])),y=phenotype[1],c=colrs[grouping],label="%d"%(phenotype[0]))
    # labeling axis in scatterplot
    plt.legend(loc="best",bbox_to_anchor=(1,1),title=" Number of \n Mutations")
    plt.ylabel("%s Level"%(pheno))
    plt.title("%s Level Based on the \nNumber of Mutations"%(pheno))
    # is save == 1 we save the scatterplot in file specified
    if(save == 1):
        plt.savefig("%s/%s_Levels_Scatterplot.png"%(file, pheno),bbox_inches="tight")
    # display the scatterplot
    plt.tight_layout()
    plt.show()

# input: string, int
# output: list of tuples
# this function inputs the mutation names and the kth grouping of f
# and outputs a list of all k combinations of gene names in lexicalgraphical order
def create_list_names(num_muts,k):
    gene_names = string.ascii_uppercase[:num_muts]
    list_names=[]
    for i in range(1,k+1):
        A = alphabet_choose_k(gene_names,i)
        B = []
        for tup in A:
            B.append(', '.join(tup))
        list_names.append(B)
    return list_names

# input: list of tuples
# output: list of tuples
# this function inputs the list of all k combinations of gene names in lexicographical order
# and outputs the indices (key) for each gene group-name in list_names
def dict_names(list_names):
    keys = range(1,len(list_names)+1)
    return (dict(zip(keys,list_names)))   

# input: list of lists, int, int
# output: vector
# this function inputs a Mallow list with specified order and kth grouping
# and outputs a single f_i mallow vector from the kth group
def Mallow_f_order(MallowList, k, order):
    for tup in MallowList:
        if (int(tup[0]))==int(k):
            list_fi= tup[1]
            return list_fi[order]

# input: vector, int
# output: list of ints
# inputs a single mallow vector and returns a list of 
# indices which contain n maximun values
def get_n_max(f_i, n_max_peak):
    n = n_max_peak
    indices = f_i.ravel().argsort()[-n:]  
    top_ind=[i for i in indices]
    top_ind.reverse()
    return top_ind

# input: vector, int
# output: list of ints
# similar function as above. Returns list of indices 
# with n minimum values
def get_n_min(f_i, n_min_peak):
    n = n_min_peak
    n2 = np.ma.masked_equal(f_i, 0.0, copy=False)
    indices = f_i.ravel().argsort()[:n]
    bot_ind=[i for i in indices]
    return bot_ind

# input: vector, int
# output: list of ints
# similar as above but works for Mallow's method on raw data
# which can have zero valued entries and if we ask for the
# minimum valued we want to exclude the zero entries because
# a zero entry means there is no data there to be analyzed
def get_n_min2(f_i, n_min_peak):
    n = n_min_peak
    f = np.ma.masked_equal(f_i, 0.0, copy=False)
    indices = f.ravel().argsort()[:n]
    bot_ind=[i for i in indices]
    return bot_ind

# input: list of tuples, list of ints, int
# output: list of tuples
# This function retrieves the peak name of max peaks or min peaks from
# a specific f^(n,k)_i.
# the first argument is the list of all k combinations of gene names 
# in lexicographical order, the second argument is a list of indeces for
# the maximum or minimum values wanted, the third argument is the order
# of the data vector from which we want the peaks
def retrieve_peak_names(diction,top_ind,order):
    order_gene_names=[]
    for i in top_ind:
        order_gene_names.append(diction.get(order)[i])
    return order_gene_names


# input: list of tuples (int, list of vectors), int, int, int, int, int, bool
# output: void
# this function inputs a Mallow list, the i'th order, the kth partition, and the num of peaks
# and outputs a bar-plot of the i'th order effects within gene groupings of k
# with the max and min peaks colored
def color_peaks(mindv, majdv, num_muts, reduct, k, order, n_max_peak, n_min_peak, save, file):
    # select correct variable to plot
    if(k >= np.floor(num_muts/2)):
        MallowDV = mindv
    else:
        MallowDV = majdv
    # define mutation names
    gene_names = string.ascii_uppercase[:num_muts]
    # all inputs from helper functions 
    f_i = Mallow_f_order(MallowDV, k, order)
    # top peaks and bottom peaks
    top_ind = get_n_max(f_i, n_max_peak)
    bot_ind = get_n_min(f_i,n_min_peak)
    # labeling peaks
    list_names = create_list_names(num_muts,k)
    diction = dict_names(list_names)
    top_peak_gene_names = retrieve_peak_names(diction, top_ind, order)
    top_min_peak_gene_names = retrieve_peak_names(diction, bot_ind, order)
   
    # the codes below changes the size and location of things
    x_axis_1 = np.arange(len(f_i))
    # width changes size of bars
    plt.bar(x_axis_1, f_i, width = 0.8, align = 'center', alpha = 0.5)
    # axis labels
    if(k >= np.floor(num_muts/2)):
        plt.xlabel('Non-Mutation Coalitions in Lexicographical Order')
    else:
        plt.xlabel('Mutation Coalitions in Lexicographical Order')
    
    plt.ylabel('Frequency')
    
    # plotting
    plt.ylim((min(f_i)-2),(max(f_i)+2))
    for i in range(len(top_ind)):
        plt.bar(top_ind[i],f_i[top_ind[i]], align='center',label=top_peak_gene_names[i])
    for j in range(len(bot_ind)):
        plt.bar(bot_ind[j],f_i[bot_ind[j]], align='center',label=top_min_peak_gene_names[j])

    # changes location of legend
    if(int(order) == 1):
        leg = 1.24
    else:
        leg = 0.02*order + 1.2
    plt.legend(bbox_to_anchor=(leg, 1), title = ' Coalitions \n Highest %d \n Lowest %d'%(n_max_peak, n_min_peak))
    
    # naming the plots
    
    if(k >= np.floor(num_muts/2)):
        plt.title('%s Reduction Data Vectors \n %d° Order Effects Within Groupings of %d Non-Mutations'%(reduct,order, k))
    else:
        plt.title('%s Reduction Data Vectors \n %d° Order Effects Within Groupings of %d Mutations'%(reduct,order, k))
    # save plot
    if(save == 1):
        plt.savefig("%s/%s_reduction_order_%d_mutation_%d.png"%(file,reduct,order,k), bbox_inches="tight")
    # show plots
    plt.show()

# input: one dimensional matrix (horizontal)
# output: tuple or int
# Takes in a row and categorizes the row by the amount of ones 
# and returns the corresponding variables to the ones 
# e.g. the row [1,0,1,0] with the columns 1234, return (1,3)
# e.g. the row [0,0,0,1] with columns 1234 returns 4
def categorize_row2(row):
    # copy row to not alter the data
    tempRow = np.copy(row)
    tempRow = np.asarray(tempRow)
    # length of row without including the last column (which are the results)
    length = len(tempRow)-1 # minus hemoglobin
    # variable to store the variables corresponding to the zeros
    category = []
    # check line for zeros
    for i in range(length):
        if(int(tempRow[i]) == 1):
            category.append(i)
    # if it is only one zero in the row it returns the int
    if(len(category) != 1):
        # returns tuple of category
        return tuple(category)
    else:
        return category[0]

# intput: matrix, int
# output:list
# Given a matrix with only nz number of zeros on each row
# returns an array of tuples where each tuple is a distinct
# combination of mutations in at least one row of the matrix
# the length of the tuples are 13-nz
def combination_table_compare(mat, nz):    
    nrows = mat.shape[0]
    nmuts = np.floor((mat.shape[1]-1)/2)
    tups = []
    for row in range(nrows):
        if(nz <= np.floor(nmuts/2)):
            tups.append(categorize_row(mat[row]))
        else:
            tups.append(categorize_row2(mat[row]))
    return list(set(tups))

# input: dataset
# output: list of lists
# Creates a list of lists with the distinct mutation combinations 
# (rows) that occur in the data, separated by amount of mutations per row.
def combinations_compare(gene_mut):
    # store raw data as a copy
    Raw_Data = np.copy(gene_mut)
    # count the numebr of variables
    num_vars = Raw_Data.shape[1]-1
    # count number of zeros on each row
    num_zeros = num_of_mutations(Raw_Data)
    # partition data by amount of zeros (list of matrices)
    Sub_Data = raw_data_partitions(Raw_Data, num_zeros)
    # list to hold combinations of mutations
    combinations_mutations = []

    # for every partition of data of the form (#zero, sub-data)
    # create a table and save it that contains its distinct
    # combinations of mutations
    for data in range(len(Sub_Data)):
        nzero = Sub_Data[data][0]
        matr = Sub_Data[data][1]
        tupslist = combination_table_compare(matr, nzero)
        combinations_mutations.append((num_vars-nzero, tupslist))
    return combinations_mutations

# input: list of ints, list of tuples
# output: bool
# Given a list of mutations and a list of tuples where each tuple is a combination of 
# mutations, it checks if the tuple is a subset of any other tuple on the list
# of tuples, if it is, it returns True, otherwise returns False
def isin_data(subtup, tuplist):
    # when k == 1 we have a list of ints instead of a list of tuples
    if(isinstance(tuplist[0],int)):
        if(len(subtup) == 1):
            if(set(subtup).issubset(set(tuplist))):
                return True
        else:
            return False
    # when k != 1
    else:
        for i in range(len(tuplist)):
            if(set(subtup).issubset(tuplist[i])):
                return True
        return False

# input: matrix, list, list, int, int, int, int, int, bool, string
# output: dataframe
# takes in the dataset and the calculated results and creates a dataframe with the
# highest and lowest values and their corresponding coalitions that actually occur
# in the data. It gives the option to save the latex code of the table into a given file
def table_values(gene_mut, mindv, majdv, num_muts, k, order, maxp, minp,save,file):
    # select correct variable to plot
    if(k >= np.floor(num_muts/2)):
        DV = mindv
    else:
        DV = majdv
    
    # min_values, max_values = find_real_fis(DV, gene_mut, num_muts)
    # capture data vector of grouping k
    for i in range(len(DV)):
        if(int(DV[i][0]) == int(k)):
            kIndex = i
    # capture the combinations of mutations that occur in the data with given k and order 
    combinations_mutations = combinations_compare(gene_mut)
    # combinations_mutations = combinations_compare(gene_mut)
    for i in range(len(combinations_mutations)):
        if(int(combinations_mutations[i][0]) == int(k)):
            combs = combinations_mutations[i][1]
    
    # capture correct data vector from results for given k and order
    real_fis = []
    for i in range(len(DV)):
        if(int(DV[i][0]) == int(k)):
            fs = DV[i][1]
    
    # create a new list containing the values and coalitions that actually occur in the data
    for fi in range(1, len(fs)):
        f = fs[fi]
        newf = []
        lexOrder = alphabet_choose_k(list(range(num_muts)), fi)
        # create new variable with the names of the coalition (more user friendly)
        lexOrderLetters = alphabet_choose_k(string.ascii_uppercase[:num_muts], fi)
        labels = []
        for tup in lexOrderLetters:
            labels.append(', '.join(tup))
        for lex in range(len(lexOrder)):
            if(isin_data(lexOrder[lex], combs)):
                newf.append( (labels[lex], round(f[lex],5)) )
        real_fis.append(newf)
    
    # get values rounded to 5 decimals
    mins = sorted(real_fis[order-1], key = itemgetter(1), reverse = False)[:minp]
    maxs = sorted(real_fis[order-1], key = itemgetter(1), reverse = True)[:maxp]
    
    # mins = min_values[:minp]
    # maxs = max_values[:maxp]

    # add NaNs to make lists have the same length
    if(len(maxs) < len(mins)):
        nans = len(mins) - len(maxs)
        for i in range(nans):
            maxs.append((None, None))
    elif(len(mins) < len(maxs)):
        nans = len(maxs) - len(mins)
        for i in range(nans):
            mins.append((None, None))

    # create dataframe with data
    dataframe = pd.DataFrame(maxs, columns=["A","B"])
    mincoal = [x[0] for x in mins]
    minval = [x[1] for x in mins]
    dataframe["C"] = pd.Series(mincoal, index=dataframe.index)
    dataframe["D"] = pd.Series(minval, index=dataframe.index)
    dataframe.columns = ['Highest', 'Values', 'Lowest', 'Values']
    
    # save tables
    if(save == 1):
        with open(file,'a') as tf:
            s = latex_table(num_muts,k,order,dataframe)
            tf.write(s)
            
    # return dataframe
    return dataframe

# input: int, int, int, dataframe
# output: string
# returns a string with latex code for a table of the given dataframe
def latex_table(num_muts, k, order, df):
    # check if we have a mutations or non-mutations dataframe
    if(k <= np.floor(num_muts/2)):
        mut1 = "Mutations"
        mut2 = "mutations"
    else:
        mut1 = "Non-Mutations"
        mut2 = "non-mutations"
    
    # create top of table
    ttable = """\n
\\begin{tabular}{|cc|cc|}
    \hline
    \multicolumn{4}{|c|}{Groupings of %d %s} \\\ 
    \hline
    \multicolumn{2}{|c|}{Highest Values} &  \multicolumn{2}{|c|}{Lowest Values} \\\ 
    \hline \hline
"""%(k,mut1)
    
    # middle of table
    mtable = ""
    
    # get dataframe values
    val = df.values
    
    # go thorugh the matrix 
    for row in range(val.shape[0]):
        # add tab at the beginning of each row or the table
        mtable += "    "
        for col in range(val.shape[1]):
            # even columns have strings
            if(col % 2 == 0):
                # for entries that are not None
                if(val[row,col]):
                    mtable += "%s & "%(val[row,col])
                # we dont write an entry on the table for None entries in the matrix
                else:
                    mtable += " & "
            # odd columns are floats
            else:
                # we dont want to write nans in our table
                if(math.isnan(val[row,col])):
                    mtable += " & "
                else:
                    mtable += "%d & "%(val[row,col])
        # eliminate the last "& " of the row
        mtable = mtable[:-2]
        # add a line in between rows
        mtable += "\\\ \hline \n"
        
    # bottom of table
    btable = "\end{tabular} \n"
    
    # return all three parts of the table as one string
    return ttable+mtable+btable

# input: matrix, list of results, int
# output: list of tuples
# checks values that actually occur in the data and returns them as lists
# of tuples where the first element of the tuple is the coalition
# and the second is the corresponding value
def find_real_fis(gene_mut, DV, k, order):
    # capture data vector of grouping k
    for i in range(len(DV)):
        if(int(DV[i][0]) == int(k)):
            kIndex = i
    # get number of mutations
    num_muts = gene_mut.shape[1]-1
    # capture the combinations of mutations that occur in the data with given k and order 
    combinations_mutations = combinations_compare(gene_mut)
    # combinations_mutations = combinations_compare(gene_mut)
    for i in range(len(combinations_mutations)):
        if(int(combinations_mutations[i][0]) == int(k)):
            combs = combinations_mutations[i][1]

    # capture correct data vector from results for given k and order
    real_fis = []
    for i in range(len(DV)):
        if(int(DV[i][0]) == int(k)):
            fs = DV[i][1]

    # create a new list containing the values and coalitions that actually occur in the data
    for fi in range(1, len(fs)):
        f = fs[fi]
        newf = []
        lexOrder = alphabet_choose_k(list(range(num_muts)), fi)
        # create new variable with the names of the coalition (more user friendly)
        lexOrderLetters = alphabet_choose_k(string.ascii_uppercase[:num_muts], fi)
        labels = []
        for tup in lexOrderLetters:
            labels.append(', '.join(tup))
        for lex in range(len(lexOrder)):
            if(isin_data(lexOrder[lex], combs)):
                newf.append( (labels[lex], lexOrder[lex], round(f[lex],5)) )
        real_fis.append(newf)
    
    # get values rounded to 5 decimals (descending)
    if(int(k) == 1):
        sortedValues = sorted(real_fis, key = itemgetter(2), reverse = True)
    else:
        sortedValues = sorted(real_fis[order-1], key = itemgetter(2), reverse = True)

    # return lists
    return sortedValues

# input: int, dataframe, string
# output: string
# returns a string with latex code for a table of the given dataframe
def latex_table2(num_muts, df, coalition):
    # create top of table
    ttable = """\n
\\begin{tabular}{|c|c|c|}
    \hline
    \multicolumn{3}{|c|}{Coalition %s} \\\ 
    \hline
    Grouping & Value & Position \\\ 
    \hline \hline
"""%(coalition)
    
    # middle of table
    mtable = ""
    
    # get dataframe values
    val = df.values
    
    # go thorugh the matrix 
    for row in range(val.shape[0]):
        # add tab at the beginning of each row or the table
        mtable += "    "
        for col in range(val.shape[1]):
            # column 1 is where the values are
            if(col == 1):
                if(val[row,col] > 0):
                    mtable += "\\textcolor{red}{%f} & "%(val[row,col])
                else:
                    mtable += "\\textcolor{blue}{%f} & "%(val[row,col])
            # the rest can be copied as is
            else:
                mtable += f"{val[row,col]} & "
                # for the last column we add the degree charater
                if(col == 2):
                    mtable = mtable[:-4]
                    mtable += "\\degree & "
                
        # eliminate the last "& " of the row
        mtable = mtable[:-2]
        # add a line in between rows
        mtable += "\\\ \hline \n"
        
    # bottom of table
    btable = "\end{tabular} \n"
    
    # return all three parts of the table as one string
    return ttable+mtable+btable


# input: matrix, int, list of lists, list of lists, string
# output: dataframe
# creates a dataframe for the values of a given coalition across different groupings of mutations
def coalition_table(gene_mut, num_muts, min_vectors, maj_vectors, coalition, save, file):
    order = len(coalition)
    # change coalition string to numbers
    coal = []
    for char in coalition:
        coal.append(ord(char)-65)
    
    ks = []
    for i in range(len(maj_vectors)):
        ks.append(maj_vectors[i][0])
    for i in range(len(min_vectors)):
        ks.append(min_vectors[i][0])

    coalition_table = []

    for k in ks:
        # search for k in correct variable
        if(k > np.floor(num_muts/2)):
            res = min_vectors
        else:
            res = maj_vectors

        # get real values of grouping k
        listLook = find_real_fis(gene_mut, res, k, order)

        # go through the list looking for a match of coalitions you are interested in
        for index in range(len(listLook)):
            if(listLook[index][1] == tuple(coal)):
                if(listLook[index][2] < 0):
                    place = len(listLook)-index
                else:
                    place = index+1
                coalition_table.append([int(k),listLook[index][2],"%d°"%(place)])

    # create dataframe with data
    dataframe = pd.DataFrame(coalition_table, columns=["Grouping","Value","Position"])
    dataframe1 = dataframe.style.applymap(color_negative_red)
    # save the latex code for table in file
    if(save == 1):
        with open(file,'a') as tf:
            s = latex_table2(num_muts, dataframe, coalition)
            tf.write(s)
    return dataframe1

    # the following function was taken from
# https://pandas.pydata.org/pandas-docs/stable/style.html
# and we altered the color scheme
def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if(isinstance(val,float)):
        color = 'blue' if val < 0 else 'red'
    else:
        color = 'black'
    return 'color: %s' % color

# the following function was taken from
# https://pandas.pydata.org/pandas-docs/stable/style.html
# and we altered the color scheme
def color_negative_red2(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if(isinstance(val,float)):
        num = str(val)
        dec = num.split('.')
        if(len(dec)>1): 
            if(len(dec[1])!=1):
                color = 'blue' if val < 0 else 'red'
            else: 
                color = 'black'
        else: 
            color = 'black'
    else:
        color = 'black'
    return 'color: %s' % color



