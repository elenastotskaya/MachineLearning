%% 
% Задание 1

clear
data = readtable("reglab1.txt")
modelX = fitlm([data.z data.y],data.x);
rSquaredX = modelX.Rsquared
modelY = fitlm([data.x data.z],data.y);
rSquaredY = modelY.Rsquared
modelZ = fitlm([data.x data.y],data.z);
rSquaredZ = modelZ.Rsquared
%% 
% Задание 2

clear
data = readtable("reglab.txt")
model1234 = fitlm([data.x1 data.x2 data.x3 data.x4],data.y);
SSE1234 = model1234.SSE
model123 = fitlm([data.x1 data.x2 data.x3],data.y);
SSE123 = model123.SSE
model124 = fitlm([data.x1 data.x2 data.x4],data.y);
SSE124 = model124.SSE
model134 = fitlm([data.x1 data.x3 data.x4],data.y);
SSE134 = model134.SSE
model234 = fitlm([data.x2 data.x3 data.x4],data.y);
SSE234 = model234.SSE
model12 = fitlm([data.x1 data.x2],data.y);
SSE12 = model12.SSE
model13 = fitlm([data.x1 data.x3],data.y);
SSE13 = model13.SSE
model14 = fitlm([data.x1 data.x4],data.y);
SSE14 = model14.SSE
model23 = fitlm([data.x2 data.x3],data.y);
SSE23 = model23.SSE
model24 = fitlm([data.x2 data.x4],data.y);
SSE24 = model24.SSE
model34 = fitlm([data.x3 data.x4],data.y);
SSE34 = model34.SSE
model1 = fitlm(data.x1,data.y);
SSE1 = model1.SSE
model2 = fitlm(data.x2,data.y);
SSE2 = model2.SSE
model3 = fitlm(data.x3,data.y);
SSE3 = model3.SSE
model4 = fitlm(data.x4,data.y);
SSE4 = model4.SSE
%% 
% Задание 3

clear
data = readtable("cygage.txt")
weightsModel = fitlm(data.Depth,data.calAge,"Weights",data.Weight);
weightsModel.SSE
weightsModel.Rsquared
%% 
% Задание 4

clear
data = readtable("longley.csv");
data.Population = []
cvpt = cvpartition(data.Employed,"HoldOut",0.5);
dataTrain = data(training(cvpt),:);
dataTest = data(test(cvpt),:);
linModel = fitlm(dataTrain);
testPrediction = predict(linModel,dataTest);
MAE = mean(abs(testPrediction-dataTest.Employed))
resubPrediction = predict(linModel,dataTrain);
MAE = mean(abs(resubPrediction-dataTrain.Employed))

lambda = 10.^(-3+0.2.*(0:25));
ridgeMAE = zeros(numel(lambda),2);
ridgeModels = ridge(dataTrain{:,end},dataTrain{:,1:end-1},lambda,0);
for i = 1:numel(lambda)
    ridgePrediction = ridgeModels(1,i) + dataTest{:,1:end-1}*ridgeModels(2:end,i);
    ridgeMAE(i,1) = mean(abs(ridgePrediction-dataTest.Employed));
    ridgeResubPrediction = ridgeModels(1,i) + dataTrain{:,1:end-1}*ridgeModels(2:end,i);
    ridgeMAE(i,2) = mean(abs(ridgeResubPrediction-dataTrain.Employed));
end
plot(lambda,ridgeMAE)
title("Ridge error")
legend('Test data','Training data')
xlabel("\lambda")
ylabel("Mean absolute error")
%% 
% Задание 5

clear
data = readtable("eustock.csv")
timestamps = 1:size(data,1);
plot(timestamps,[data.DAX data.SMI data.CAC data.FTSE])
legend('DAX','SMI','CAC','FTSE',"Location","northwest")
modelDAX = fitlm(timestamps,data.DAX);
modelDAX.SSE
modelDAX.Rsquared
modelDAX.Coefficients

modelSMI = fitlm(timestamps,data.SMI);
modelSMI.SSE
modelSMI.Rsquared
modelSMI.Coefficients

modelCAC = fitlm(timestamps,data.CAC);
modelCAC.SSE
modelCAC.Rsquared
modelCAC.Coefficients

modelFTSE = fitlm(timestamps,data.FTSE);
modelFTSE.SSE
modelFTSE.Rsquared
modelFTSE.Coefficients

% modelFull = fitlm(timestamps,[data.DAX data.SMI data.CAC data.FTSE]);
% modelFull.SSE
% modelFull.Rsquared
%% 
% Задание 6

clear
data = readtable("JohnsonJohnson.csv")
data.year = zeros(size(data,1),1);
data.q = zeros(size(data,1),1);
index = 1;
for year = 1960:1980
    for q = 1:4
        data.year(index) = year;
        data.q(index) = q;
        index = index + 1;
    end
end
plot(data.year(data.q == 1),[data.value(data.q == 1) data.value(data.q == 2) data.value(data.q == 3) data.value(data.q == 4)])
legend('Q1','Q2','Q3','Q4',"Location","northwest")

modelQ1 = fitlm(data.year(data.q == 1),data.value(data.q == 1));
modelQ1.Coefficients
modelQ2 = fitlm(data.year(data.q == 2),data.value(data.q == 2));
modelQ2.Coefficients
modelQ3 = fitlm(data.year(data.q == 3),data.value(data.q == 3));
modelQ3.Coefficients
modelQ4 = fitlm(data.year(data.q == 4),data.value(data.q == 4));
modelQ4.Coefficients

fullModel = fitlm([data.year data.q],data.value);
pred2016 = zeros(4,1);
pred2016(1) = predict(fullModel,[2016 1]);
pred2016(2) = predict(fullModel,[2016 2]);
pred2016(3) = predict(fullModel,[2016 3]);
pred2016(4) = predict(fullModel,[2016 4])
mean2016 = mean(pred2016)
%% 
% Задание 7

clear
data = readtable("cars.csv")
model = fitlm(data);
prediction = predict(model,40)
%% 
% Задание 8

clear
data = readtable("svmdata6.txt")
cvpt = cvpartition(data.Var3,"HoldOut",0.3);
dataTrain = data(training(cvpt),:);
dataTest = data(test(cvpt),:);
epsilon = 0.1:0.1:1;
modelMSE = zeros(numel(epsilon),1);
for i = 1:numel(epsilon)
    svmModel = fitrsvm(dataTrain,"Var3","KernelFunction","rbf","Epsilon",epsilon(i));
    prediction = predict(svmModel,dataTest);
    modelMSE(i) = mean((prediction - dataTest.Var3).^2);
end
plot(epsilon,modelMSE)
legend('epsilon','MSE')
%% 
% Задание 9

clear
data = readtable("nsw74psid1.csv")
cvpt = cvpartition(data.re78,"HoldOut",0.3);
dataTrain = data(training(cvpt),:);
dataTest = data(test(cvpt),:);

treeModel = fitrtree(dataTrain,"re78");
prediction = predict(treeModel,dataTest);
treeMSE = mean((prediction - dataTest.re78).^2)

linModel = fitlm(dataTrain);
prediction = predict(linModel,dataTest);
linMSE = mean((prediction - dataTest.re78).^2)

svmModel = fitrsvm(dataTrain,"re78");
prediction = predict(svmModel,dataTest);
svmMSE = mean((prediction - dataTest.re78).^2)