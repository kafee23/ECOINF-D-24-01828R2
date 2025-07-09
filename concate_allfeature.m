clc
clear

%% load the trining dataset LDP 3x3

dataSetLDP=load('featureMat_Anth3_LDP.mat');
trainingSetLDP_Anth=dataSetLDP.featureMat_Anth_LDP; 


%% normalize LDP 3x3

maxValue=max(max(trainingSetLDP_Anth));
trainingSetLDP_Anth_N=trainingSetLDP_Anth/maxValue;


%% load the trining dataset LDP 6x6

dataSetLDP=load('featureMat_Anth6_LDP.mat');
trainingSetLDP_Anth6=dataSetLDP.featureMat_Anth6_LDP; 


%% normalize LDP 6x6

maxValue=max(max(trainingSetLDP_Anth6));
trainingSetLDP_Anth6_N=trainingSetLDP_Anth6/maxValue;



%% load the trining dataset LDP 9x9

dataSetLDP=load('featureMat_Anth9_LDP.mat');
trainingSetLDP_Anth9=dataSetLDP.featureMat_Anth9_LDP; 


%% normalize LDP 9x9

maxValue=max(max(trainingSetLDP_Anth9));
trainingSetLDP_Anth9_N=trainingSetLDP_Anth9/maxValue;


%% load the trining dataset LDPv 3x3

dataSetLDPv=load('featureMat_Anth3_LDPv.mat');
trainingSetLDPv_Anth=dataSetLDPv.featureMat_Anth_LDPv; 


%% normalize LDPv 3x3

maxValue=max(max(trainingSetLDPv_Anth));
trainingSetLDPv_Anth_N=trainingSetLDPv_Anth/maxValue;


%% load the trining dataset LDPv 6x6

dataSetLDPv=load('featureMat_Anth6_LDPv.mat');
trainingSetLDPv_Anth6=dataSetLDPv.featureMat_Anth6_LDPv; 


%% normalize LDPv 6x6

maxValue=max(max(trainingSetLDPv_Anth6));
trainingSetLDPv_Anth6_N=trainingSetLDPv_Anth6/maxValue;



%% load the trining dataset LDPv 9x9

dataSetLDPv=load('featureMat_Anth9_LDPv.mat');
trainingSetLDPv_Anth9=dataSetLDPv.featureMat_Anth9_LDPv; 


%% normalize LDPv 9x9

maxValue=max(max(trainingSetLDPv_Anth9));
trainingSetLDPv_Anth9_N=trainingSetLDPv_Anth9/maxValue;


%% Concatanation After Normalization
trainingSet_Anth_N=[trainingSetLDPv_Anth_N trainingSetLDPv_Anth6_N trainingSetLDPv_Anth9_N trainingSetLDP_Anth_N trainingSetLDP_Anth6_N trainingSetLDP_Anth9_N];



%% Concatanation Before Normalization
trainingSet_Anth=[trainingSetLDPv_Anth trainingSetLDPv_Anth6 trainingSetLDPv_Anth9 trainingSetLDP_Anth trainingSetLDP_Anth6 trainingSetLDP_Anth9];



