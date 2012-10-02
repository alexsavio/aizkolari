
#------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#2012-06-18
#------------------------------------------------------------------------------

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import os
import numpy as np
import numpy.random as rand
import aizkolari_utils as au
from cvpartition import cvpartition

def targets_to_classidx (targets, classes):
#given class labels targets Nx1 and 
#a vector with all the existing classes Sx1 of the problem
#returns a matrix SxN where 
#[i,j] is 1 if target j is of class i, -1 otherwise
    classes.sort()
    classidx = np.zeros([len(classes), len(targets)], dtype=int)
    for i in np.arange(len(classes)):
        c = classes[i]
        classidx[i, targets == c] = 1
    classidx = classidx * 2 - 1

    return classidx

#hidden layer
def hidden_layer (data, inputweight, hidneurons_bias, act_function):
#data: NxD
#inputweight: HxD
#hidneurons_bias: array with bias of hidden neurons
#act_function: name of activation function: 'sig', 'sin', 'hardlim' or 'radbas'
#returns: HxN

    data_size = data.shape[0]
    cell_num  = len(hidneurons_bias)

    H = np.dot(inputweight, data.transpose())

    bias_matrix = hidneurons_bias.reshape([1,cell_num]).repeat(data_size, axis=0).transpose()

    H = H + bias_matrix

    # Calculate hidden neuron output matrix H
    if act_function == 'sig' or act_function == 'sigmoid':
        #sigmoid
        H = 1 / (1 + np.exp(-H))
    elif act_function == 'sin' or act_function == 'sine':
        # Sine
        H = np.sin(H)
    elif act_function == 'hardlim':
        # Hard Limit
        H = (H >= 0) * 1.
    elif act_function == 'radbas':
        # Radial basis function
        H = np.exp(-np.power(H,2))
    #elif act_function == 'tribas':
        # Triangular basis function
    #    H = tribas(tempH);
    return H

def myloo (x,y): function [w,yloo,errloo]=myloo(x,y)
    [N,d] = x.shape
    w = np.dot(np.linalg.pinv(x), y)
    P = np.dot(x, np.linalg.inv(np.dot(x.transpose(), x)));
    mydiag = np.dot(P,x,2);
S=(y-x*w)./(1-mydiag);
errloo=mean(S.^2);
yloo=y-S;
    return w,yloo,errloo

def elm_classifier (traindata_file, trainlabs_file, testdata_file, testlabs_file, hidneurons_num, act_function):
#function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)

# Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
# OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)

# Input:
# TrainingData_File     - Filename of training data set
# TestingData_File      - Filename of testing data set
# TrainingLabel File    - Filename of training classification target labels
# TestingLabel File     - Filename of testing classification target labels
#Data files should be Numpy compatible with NxD matrices, where N is the 
# number of subjects and D the dimensions of each subject_data_slicing
#Labels files shoule will be loaded with loadtxt Numpy function, should have
# one label [-1 or 1] (in binary classification) for each subject, in each line
# and should be in the same order as in the data files.

# Elm_Type              - 0 for regression; 1 for (both binary and 
#                                           multi-classes) classification
# NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
# ActivationFunction    - Type of activation function:
#                           'sig' for Sigmoidal function
#                           'sin' for Sine function
#                           'hardlim' for Hardlim function
#                           'tribas' for Triangular basis function
#                           'radbas' for Radial basis function 
#                     (for additive type of SLFNs instead of RBF type of SLFNs)

# Output: 
# TrainingTime      eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee    - Time (seconds) spent on training ELM
# TestingTime           - Time (seconds) spent on predicting ALL testing data
# TrainingAccuracy      - Training accuracy: 
#                           RMSE for regression or correct classification rate for classification
# TestingAccuracy       - Testing accuracy: 
#                           RMSE for regression or correct classification rate for classification
#
# MULTI-CLASS CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
# FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
# neurons; neuron 5 has the highest output means input belongs to 5-th class
#
# Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
# Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')

    #Matlab version
    #    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    #    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    #    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    #    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    #    DATE:       APRIL 2004

    #hidneurons_num = 10
    #act_function   = 'sig'

    #ELM type
    REGRESSION, CLASSIFIER = range(2)

    elm_type       = CLASSIFIER
    timer          = au.get_timer()

    #import os
    #rootdir        = '/media/oasis_post/cv_jacs/pearson_0001'
    #rootdir        = '/scratch/cv_niftiseg_gm/pearson_0001'
    #traindata_file = rootdir + os.path.sep + 'pearson_90thrP_features.scaled.npy'
    #trainlabs_file = rootdir + os.path.sep + 'subjlabels_included'
    #testdata_file  = rootdir + os.path.sep + 'pearson_90thrP_excludedfeats.scaled.npy'
    #testlabs_file  = rootdir + os.path.sep + 'subjlabels_excluded'

    # Load training dataset, should be NxD, N labels
    traindata = np.load   (traindata_file)
    trainlabs = np.loadtxt(trainlabs_file)

    #Load testing dataset, should be NxD, N labels
    testdata = np.load   (testdata_file)
    testlabs = np.loadtxt(testlabs_file)

    #configuring settings from data
    traindata_size = traindata.shape[0]
    testdata_size  = testdata.shape [0]
    inneurons_num  = traindata.shape[1]

    if elm_type != REGRESSION:
        #Preprocessing the data of classification
        classes = np.unique(np.array([np.unique(trainlabs), np.unique(testlabs)]))
        class_num = len(classes)

        outneurons_num = class_num

        #Processing the targets of training
        train_T = targets_to_classidx (trainlabs, classes)

        #Processing the targets of testing
        test_T = targets_to_classidx (testlabs, classes)

    # Calculate weights & biases
    start_time_train = timer();

    # Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
    inweight = rand.random([hidneurons_num, inneurons_num]) * 2 - 1
    hidneurons_bias = rand.random(hidneurons_num)

    train_H = hidden_layer (traindata, inweight, hidneurons_bias, act_function)

    # Calculate output weights OutputWeight (beta_i)
    #method 1
    outweights = np.dot(np.linalg.pinv(train_H.transpose()), train_T.transpose())

    #faster method 2
    C = 1
    #outweights = np.dot(np.linalg.inv((np.eye(H.shape[0])/C) + (np.dot (H,H.transpose()))), np.dot(H, trainlabs))
    #implementation; one can set regularizaiton factor C properly in classification applications 

    #faster method 2
    #outweights = np.solve(np.eye(H.shape[0])/C + np.dot(H, H.transpose()), np.dot(H, trainlabs))
    #implementation; one can set regularizaiton factor C properly in classification applications

    #If you use faster methods or kernel method, PLEASE CITE in your paper properly: 
    #Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010. 

    end_time_train = timer()

    training_time = end_time_train - start_time_train

    #Calculate the training accuracy
    #Y: the actual output of the training data
    train_Y = np.dot(train_H.transpose(), outweights).transpose()

    if elm_type == REGRESSION:
        #Calculate training accuracy (RMSE) for regression case
        training_accuracy = np.sqrt(np.mean(np.sum(np.power((train_T - Y),2), axis=0))/train_T.shape[0]
        print('Training accuracy: ' + str(training_accuracy))

    #Calculate the output of testing input
    start_time_test = timer()

    test_H = hidden_layer (testdata, inweight, hidneurons_bias, act_function)

    #   TY: the actual output of the testing data
    test_Y = np.dot(test_H.transpose(), outweights).transpose()

    end_time_test = timer()

    #CPU time (seconds) spent by ELM predicting the whole testing data
    testing_time = end_time_test - start_time_test

    if elm_type == REGRESSION:
        testing_accuracy = np.sqrt(np.mean(np.sum(np.power((test_T - Y),2), axis=0))/test_T.shape[0]

        return training_accuracy, testing_accuracy

    elif elm_type == CLASSIFIER:
        train_Ylab = np.argmax(train_Y, axis=0)
        train_Tlab = np.argmax(train_T, axis=0)

        test_Ylab = np.argmax(test_Y, axis=0)
        test_Tlab = np.argmax(test_T, axis=0)

        if class_num > 2:
            # Calculate training & testing classification accuracy
            missrate_train = 0;
            missrate_test  = 0;

            #training miss classification rate
            missrate_train = np.sum(train_Tlab != train_Ylab)
            missrate_train = 1 - missrate_train/traindata_size

            #testing miss classification rate
            missrate_test = np.sum(test_Tlab != test_Ylab)
            missrate_test = 1 - missrate_test/testdata_size

            return missrate_train, missrate_train

        #the binary classification case
        elif class_num == 2:
            #tp, fp, tn, fn
            confmat_train = np.zeros(4, dtype=int)
            confmat_test  = np.zeros(4, dtype=int)

            confmat_train[0] = np.sum (train_Tlab[train_Tlab == 1] == train_Ylab[train_Tlab == 1])
            confmat_train[1] = np.sum (train_Tlab[train_Tlab == 0] != train_Ylab[train_Tlab == 0])
            confmat_train[2] = np.sum (train_Tlab[train_Tlab == 0] == train_Ylab[train_Tlab == 0])
            confmat_train[3] = np.sum (train_Tlab[train_Tlab == 1] != train_Ylab[train_Tlab == 1])

            confmat_test [0] = np.sum (test_Tlab [test_Tlab == 1]  == test_Ylab [test_Tlab  == 1])
            confmat_test [1] = np.sum (test_Tlab [test_Tlab == 0]  != test_Ylab [test_Tlab  == 0])
            confmat_test [2] = np.sum (test_Tlab [test_Tlab == 0]  == test_Ylab [test_Tlab  == 0])
            confmat_test [3] = np.sum (test_Tlab [test_Tlab == 1]  != test_Ylab [test_Tlab  == 1])

            return confmat_train, confmat_test



def elmmc (data, labels):
#function [bestacc,yloo,Ht,m,s,W1t,W10t,w]=ELMMC(x,y)
# data is NxD
# labels is NxNC, where NC is the number of classes where labels[i,j] is 1 if 
# subject i belongs to class indexed as j, 0 the rest.
    [N, d] = data.shape

    [N, NC] = labels.shape

    #normalizing data
    m    = np.tile(np.mean(data, axis=0), [N, 1])
    s    = np.tile(np.std (data, axis=0), [N, 1])
    data = (data - m)/s


    
Ht=[];
W1t=[];
W10t=[];
dkk=1;
dkk1=1;
bestacc=0;

    nn = 1
    dk = 1
    while dk < 40:
        acc = 0
        W1new  = np.random.randn(d,nn) * np.sqrt(3)
        W10new = np.random.randn(1,nn) * np.sqrt(3)
        Hnew   = np.tanh(np.dot(data, W1new) + np.dot(np.ones([N,1]), W10new));

        Hnew   = np.tanh(np.dot(data_mat, W1new_mat) + np.dot(np.ones([N,1]), W10new_mat))

        if Ht:
            Htnew = np.concatenate ((Ht, Hnew), axis=1)
        else:
            Htnew = Hnew

        for i in np.arange(NC):
            [A, ]

        for i=1:NC
            [A,yloo(:,i),~]=myloo(Htnew,y(:,i));
        end
        [I,IItrue]=max(y,[],2);
        [I,II]=max(yloo,[],2);
        accnew=mean(II==IItrue);
        if accnew>acc
            acc=accnew;
            H=Hnew;
            W1=W1new;
            W10=W10new;
        end
    end
    
    if acc>bestacc
        bestacc=acc;
        Ht=[Ht H];
        W1t=[W1t W1];
        W10t=[W10t W10];
        dkk1=dkk1+1;
        dkk;
        dkk1;
    end
    dkk=dkk+1;
end

for i=1:NC
    [A,yloo(:,i),~]=myloo(Ht,y(:,i));
end
[I,yh]=max(yloo,[],2);
w=Ht\y;

yloo=zeros(N,NC);
for i=1:N
    yloo(i,yh(i))=1;
end




function [w,yloo,errloo]=myloo(x,y)

[N,d]=size(x);
w=pinv(x)*y;
P=x*inv(x'*x);
mydiag=dot(P,x,2);
S=(y-x*w)./(1-mydiag);
errloo=mean(S.^2);
yloo=y-S;

def fitness_OPELM(data, labels, population, nrepeat, cv_foldn=10)
#for binary classification where labels are -1 or 1
#data is NxD, where N is the total number of subjects
#population is a genetic algorithm generation GxD of 1s and 0s that selects features from D
#nrepeat is the number of times to repeat the fold cv_foldn cross-validation
#returns  [Accuracy, Sensitivity, Specificity, Accuracy_detail, Sensitivity_detail, Specificity_detail]
    popsize  = len(population)
    datasize = data.shape[0]
    for j in np.arange(popsize):
        indiv  = population[j,:]
        index  = indiv == 1
        x      = data[:,indx]

        controls = x[labels == -1, :]
        patients = x[labels ==  1, :]

        ncontrols = controls.shape[0]
        npatients = patients.shape[0]

        #these are Nxfoldn
        cnt_cv_idx = cvpartition(ncontrols, cv_foldn);
        pat_cv_idx = cvpartition(npatients, cv_foldn);

bestacc_cv=[];
yloo_cv=[];
Ht_cv=[];
m_cv=[];
s_cv=[];
W1t_cv=[];
W10t_cv=[];
w_cv=[];
ytesth_cv=[];
acctest_cv=[];
Accuracy_cv=[];
Sensitivity_cv=[];
Specificity_cv=[];
Valid=[];
TN1=[];
FP1=[];
TP1=[];
FN1=[];
TN=[];
FP=[];
TP=[];
FN=[];
bestacc_train_mean=[];
acctest_test_mean=[];
Accuracy_cv_mean=[];
Sensitivity_cv_mean=[];
Specificity_cv_mean=[];
Accuracy_all=[];

    for r in np.arange(nrepeat):
        for f in np.arange(cv_foldn):
            #train data
            traincnts = controls[cnt_cv_idx[:,f] == False, :]
            trainpats = patients[pat_cv_idx[:,f] == False, :]

            traincntslabs = np.ones(traincnts.shape[0], dtype=int) * -1
            trainpatslabs = np.ones(trainpats.shape[0], dtype=int)

            traindata = np.concatenate ((traincnts,trainpats))
            tlabs     = np.concatenate ((traincntslabs,trainpatslabs))
            trainlabs = np.zeros([len(tlabs), 2], dtype=int)
            trainlabs[tlabs == -1, 0] = 1
            trainlabs[tlabs ==  1, 1] = 1

            #test data
            testcnts = controls[cnt_cv_idx[:,f] == True, :]
            testpats = patients[pat_cv_idx[:,f] == True, :]

            testcntslabs = np.ones(testcnts.shape[0], dtype=int) * -1
            testpatslabs = np.ones(testpats.shape[0], dtype=int)

            testdata = np.concatenate ((testcnts,testpats))
            tlabs    = np.concatenate ((testcntslabs,testpatslabs))
            testlabs = np.zeros([len(tlabs), 2], dtype=int)
            testlabs[tlabs == -1, 0] = 1
            testlabs[tlabs ==  1, 1] = 1

            #OP-ELM
            #train
            [bestacc,yloo,Ht,m,s,W1t,W10t,w] = elmmc (traindata, trainlabs)

   bestacc_cv(end+1)=bestacc;
   %H_cv(end+1)=H;   
   m_cv(:,end+1)=m;   
   s_cv(:,end+1)=s; 
   %W1(end+1)=W1;   
   %W10_cv(:,end+1)=W10;   
   %w_cv(:,end+1)=w;
   %test
   %[ytesth,acctest]=ELMtest_DARYA(x_test,label_test,m,s,W1,W10,w);
   [ytesth,acctest]=ELMMCtest(x_test,label_test,m,s,W1t,W10t,w);
   %ytesth_cv(:,end+1)=ytesth;
   acctest_cv(end+1)=acctest;
   
   %accuracy as in our articles
   % para controles
    for ksi=1:n_label_test_con
        if ytesth(ksi,1)==1
            Valid(end+1)=1;
            TN1(end+1)=1;
            FP1(end+1)=0;
        else FP1(end+1)=1;
            TN1(end+1)=0;
        end       
    end
    
    TN=sum(TN1(:));
    FP=sum(FP1(:));
    
     % para pacientes
    for ksi=n_label_test_con+1:n_label_test_con+n_label_test_pat
        if ytesth(ksi,1)==0
           Valid(end+1)=1;
           TP1(end+1)=1;
           FN1(end+1)=0;
        else FN1(end+1)=1;
           TP1(end+1)=0;
        end       
    end
    TP=sum(TP1(:));
    FN=sum(FN1(:));   

    % Accuracy
    if isempty(Valid)==0
        Accuracy1=sum(Valid(:))*100/(n_label_test_con+n_label_test_pat);
    else Accuracy1=0
    end
    
    % Sensitivity
      Sensitivity1=(TP)/(TP+FN);
    % Specificity
      Specificity1=(TN)/(TN+FP);
      
      Accuracy_cv(end+1)=Accuracy1;
      Sensitivity_cv(end+1)=Sensitivity1;
      Specificity_cv(end+1)=Specificity1;
      Accuracy_all(end+1)=Accuracy1(:);
    
end

bestacc_train_mean(end+1)=mean(bestacc_cv(:));
acctest_test_mean(end+1)=mean(acctest_cv(:));
Accuracy_cv_mean(end+1)=mean(Accuracy_cv(:));
Sensitivity_cv_mean(end+1)=mean(Sensitivity_cv(:));
Specificity_cv_mean(end+1)=mean(Specificity_cv(:));

end


Accuracy(j)=mean(Accuracy_cv(:));
Sensitivity(j)=mean(Sensitivity_cv(:));
Specificity(j)=mean(Specificity_cv(:));

Accuracy_detail(j,:)=Accuracy_cv(:);
Sensitivity_detail(j,:)=Sensitivity_cv(:);
Specificity_detail(j,:)=Specificity_cv(:);

end

%Accuracy=mean(Accuracy_cv_mean(:));
%Sensitivity=mean(Sensitivity_cv_mean(:));
%Specificity=mean(Specificity_cv_mean(:));


%Acc_OP=mean(acctest_cv(:))
%Acc_mean=mean(Accuracy_cv(:))
%Sen_mean=mean(Sensitivity_cv(:))
%Spe_mean=mean(Specificity_cv(:))


