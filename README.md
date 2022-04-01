A re-implementation of MGCN

## Environment

* python 3.9.7
* pytorch 1.10.1

## Instructions

1 Before running you program, the path of the dataset may need to be modified on your computer,
  and you may need to modify some values of some variables according to your data set in utils.py
2 Split multi-pair shot dataset and the query dataset:
    run data_split_csv.py
3 For open set dataset, please follow these steps to perform the program:
(1) run train_step1.py (pretrain Conv4)
(2) run train_step2.py (train MGCN)
(3) run test_noDA.py (test MGCN without DA) 
(4) run test_DA.py (test MGCN with DA) 

You can test the classification accuracy in different situations by modifying the relevant parameters
train-way,test-way,shot



