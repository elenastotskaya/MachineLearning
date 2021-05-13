%% 
% Задание 1

clear
data0 = readtable('nn_0.csv')
cvpt = cvpartition(data0.class,"HoldOut",0.3);
data0_train = table2array(data0(training(cvpt),:));
data0_test = table2array(data0(test(cvpt),:));
XTrain = data0_train(:,1:2);
XTest = data0_test(:,1:2);
resultTrain = categorical(data0_train(:,3));
resultTest = categorical(data0_test(:,3));
layers = [featureInputLayer(2),fullyConnectedLayer(2),softmaxLayer,classificationLayer];
options = trainingOptions('adam','MaxEpochs',50,"Plots","training-progress");
net = trainNetwork(XTrain,resultTrain,layers,options);
prediction = classify(net,XTest);
confusionchart(resultTest,prediction)

data1 = readtable('nn_1.csv')
cvpt = cvpartition(data1.class,"HoldOut",0.3);
data1_train = cell2mat(table2cell(data1(training(cvpt),:)));
data1_test = cell2mat(table2cell(data1(test(cvpt),:)));
XTrain = data1_train(:,1:2);
XTest = data1_test(:,1:2);
resultTrain = categorical(data1_train(:,3));
resultTest = categorical(data1_test(:,3));
layers = [featureInputLayer(2),fullyConnectedLayer(2),softmaxLayer,classificationLayer];
options = trainingOptions('adam','MaxEpochs',50,"Plots","training-progress");
net = trainNetwork(XTrain,resultTrain,layers,options);
prediction = classify(net,XTest);
confusionchart(resultTest,prediction)
%% 
% Задание 2

clear
data = readtable('nn_1.csv')
cvpt = cvpartition(data.class,"HoldOut",0.3);
data_train = table2array(data(training(cvpt),:));
data_test = table2array(data(test(cvpt),:));
XTrain = data_train(:,1:2);
XTest = data_test(:,1:2);
resultTrain = categorical(data_train(:,3));
resultTest = categorical(data_test(:,3));
layers = [featureInputLayer(2),fullyConnectedLayer(10),reluLayer,fullyConnectedLayer(2),softmaxLayer,classificationLayer];
options = trainingOptions('adam','MaxEpochs',50,'Plots','training-progress','Shuffle','every-epoch','InitialLearnRate',0.1);
net = trainNetwork(XTrain,resultTrain,layers,options);
prediction = classify(net,XTest);
confusionchart(resultTest,prediction)
%% 
% Задание 3

clear
dsMNISTTrain = imageDatastore("MNIST - JPG - training","IncludeSubfolders",true,"LabelSource","foldernames");
dsMNISTTest = imageDatastore("MNIST - JPG - testing","IncludeSubfolders",true,"LabelSource","foldernames");
layers = [imageInputLayer([28 28 1]),fullyConnectedLayer(10),reluLayer,fullyConnectedLayer(10),reluLayer,fullyConnectedLayer(10),softmaxLayer,classificationLayer];
options = trainingOptions('adam',"MaxEpochs",5,"Plots","training-progress");
net = trainNetwork(dsMNISTTrain,layers,options);
%%
prediction = classify(net,dsMNISTTest);
confusionchart(dsMNISTTest.Labels,prediction)