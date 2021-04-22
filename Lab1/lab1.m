%% 
% Задание 1

clear
tictactoe = readtable("tic_tac_toe.txt")
spam = readtable("spam.csv")
%%
plotDependence(tictactoe, "mvmn");
plotDependence(spam,"kernel");
%% 
% Задание 2

clear
N = 100;
Nminus = 10;
Nplus = N - Nminus;

M1minus = 15;
M2minus = 18;
Dminus = 4;
M1plus = 18;
M2plus = 18;
Dplus = 2;

X1 = zeros(N,1);
X1(1:Nminus) = randn(Nminus,1).*sqrt(Dminus) + M1minus;
X1(Nminus+1:N) = randn(Nplus,1).*sqrt(Dplus) + M1plus;

X2 = zeros(N,1);
X2(1:Nminus) = randn(Nminus,1).*sqrt(Dminus) + M2minus;
X2(Nminus+1:N) = randn(Nplus,1).*sqrt(Dplus) + M2plus;

results = ones(N,1);
results(1:Nminus) = -1;

data = table(X1,X2,results,'VariableNames',{'X1','X2','Class'})
gscatter(data.X1,data.X2,data.Class)
%%
cvpt = cvpartition(data.Class,"HoldOut",0.4);
dataTrain = data(training(cvpt),:);
dataTest = data(test(cvpt),:);
nbModel = fitcnb(dataTrain,"Class");
[prediction, scores] = predict(nbModel,dataTest);

accuracy = nnz(prediction == dataTest{:,end})/numel(prediction)
confusionchart(dataTest{:,end},prediction)

[X,Y,T,AUC] = perfcurve(dataTest.Class,scores(:,1),'-1');
AUC
plot(X,Y)
title('ROC curve')
xlabel('False positive rate') 
ylabel('True positive rate')
[X,Y] = perfcurve(dataTest.Class,scores(:,1),'-1','XCrit','prec','YCrit','reca');
plot(X,Y)
xlabel('Precision')
ylabel('Recall')
title('PR curve')

%% 
% Задание 3

clear
glass = readtable('glass.csv')
glass = glass(:,2:end);
cvpt = cvpartition(glass.Type,"HoldOut",0.2);
dataTrain = glass(training(cvpt),:);
dataTest = glass(test(cvpt),:);
maxNum = 50;
classError = zeros(maxNum,1);
for i = 1:maxNum
    knnModel = fitcknn(dataTrain,"Type","NumNeighbors",i);
    prediction = predict(knnModel,dataTest);
    classError(i) = nnz(prediction ~= dataTest{:,end})/numel(prediction);
end
plot(1:maxNum, classError)

metricTypes = ["cityblock" "chebychev" "euclidean" "minkowski"];
classError = zeros(numel(metricTypes),1);
count = 1;
for metric = metricTypes
    knnModel = fitcknn(dataTrain,"Type","NumNeighbors",5,"Distance",metric);
    prediction = predict(knnModel,dataTest);
    classError(count) = nnz(prediction ~= dataTest{:,end})/numel(prediction);
    count = count + 1;
end
plot(1:numel(metricTypes), classError)
hold on
h = zeros(4,1);
for i = 1:4
    h(i) = plot(i,classError(i),'o');
end
title('Comparison of distance metrics')
legend(h,metricTypes,'Location','Southeast');
hold off

knnModel = fitcknn(dataTrain,"Type","NumNeighbors",5);

predictionSingle = predict(knnModel, [1.516 11.7 1.01 1.19 72.59 0.43 11.44 0.02 0.1])
%% 
% Задание 4a

clear
svmdataA = readtable('svmdata_a.txt')
svmdataA = svmdataA(:,2:end);
svmdataA_test = readtable('svmdata_a_test.txt')
svmdataA_test = svmdataA_test(:,2:end);
svmModel = fitcsvm(svmdataA,"Var4","KernelFunction","linear")

supportVectorCount = nnz(svmModel.IsSupportVector)

testPrediction = predict(svmModel,svmdataA_test);
resubPrediction = resubPredict(svmModel);
confusionchart(svmdataA_test{:,end},testPrediction)
confusionchart(svmdataA{:,end},resubPrediction)

[x1Grid,x2Grid] = meshgrid(linspace(min(svmdataA_test{:,1}),max(svmdataA_test{:,1}),100),...
    linspace(min(svmdataA_test{:,2}),max(svmdataA_test{:,2}),100));
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);

plotSVM(svmdataA_test,scores,x1Grid,x2Grid,'Dataset A, linear kernel');
%% 
% Задание 4b

clear
svmdataB = readtable('svmdata_b.txt')
svmdataB = svmdataB(:,2:end);
svmdataB_test = readtable('svmdata_b_test.txt')
svmdataB_test = svmdataB_test(:,2:end);

constraintValues = [1, 10, 100, 100, 1000, 10000];
testAccuracy = zeros(1,length(constraintValues));
resubAccuracy = zeros(1,length(constraintValues));
accIndex = 1;
for C = constraintValues
    svmModel = fitcsvm(svmdataB,"Var4","KernelFunction","linear","BoxConstraint",C);
    testPrediction = predict(svmModel,svmdataB_test);
    resubPrediction = resubPredict(svmModel);
    testAccuracy(accIndex) = nnz(string(testPrediction) == string(svmdataB_test{:,end}))/numel(testPrediction);
    resubAccuracy(accIndex) = nnz(string(resubPrediction) == string(svmdataB{:,end}))/numel(resubPrediction);
    accIndex = accIndex + 1;
end
testAccuracy
resubAccuracy
%% 
% Задание 4c

clear
svmdataC = readtable('svmdata_c.txt')
svmdataC = svmdataC(:,2:end);
svmdataC_test = readtable('svmdata_c_test.txt')
svmdataC_test = svmdataC_test(:,2:end);

[x1Grid,x2Grid] = meshgrid(linspace(min(svmdataC_test{:,1}),max(svmdataC_test{:,1}),100),...
    linspace(min(svmdataC_test{:,2}),max(svmdataC_test{:,2}),100));
svmModel = fitcsvm(svmdataC,"Var4","KernelFunction","linear");
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataC_test,scores,x1Grid,x2Grid,'Linear kernel');
svmModel = fitcsvm(svmdataC,"Var4","KernelFunction","polynomial","PolynomialOrder",1);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataC_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 1');
svmModel = fitcsvm(svmdataC,"Var4","KernelFunction","polynomial","PolynomialOrder",2);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataC_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 2');
svmModel = fitcsvm(svmdataC,"Var4","KernelFunction","polynomial","PolynomialOrder",3);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataC_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 3');
svmModel = fitcsvm(svmdataC,"Var4","KernelFunction","polynomial","PolynomialOrder",4);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataC_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 4');
svmModel = fitcsvm(svmdataC,"Var4","KernelFunction","polynomial","PolynomialOrder",5);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataC_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 5');
svmModel = fitcsvm(svmdataC,"Var4","KernelFunction","gaussian");
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataC_test,scores,x1Grid,x2Grid,'Gaussian kernel');
%% 
% Задание 4d

clear
svmdataD = readtable('svmdata_d.txt')
svmdataD = svmdataD(:,2:end);
svmdataD_test = readtable('svmdata_d_test.txt')
svmdataD_test = svmdataD_test(:,2:end);

[x1Grid,x2Grid] = meshgrid(linspace(min(svmdataD_test{:,1}),max(svmdataD_test{:,1}),100),...
    linspace(min(svmdataD_test{:,2}),max(svmdataD_test{:,2}),100));
svmModel = fitcsvm(svmdataD,"Var4","KernelFunction","linear");
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataD_test,scores,x1Grid,x2Grid,'Linear kernel');
svmModel = fitcsvm(svmdataD,"Var4","KernelFunction","polynomial","PolynomialOrder",1);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataD_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 1');
svmModel = fitcsvm(svmdataD,"Var4","KernelFunction","polynomial","PolynomialOrder",2);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataD_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 2');
svmModel = fitcsvm(svmdataD,"Var4","KernelFunction","polynomial","PolynomialOrder",3);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataD_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 3');
svmModel = fitcsvm(svmdataD,"Var4","KernelFunction","polynomial","PolynomialOrder",4);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataD_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 4');
svmModel = fitcsvm(svmdataD,"Var4","KernelFunction","polynomial","PolynomialOrder",5);
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataD_test,scores,x1Grid,x2Grid,'Polynomial kernel, order = 5');
svmModel = fitcsvm(svmdataD,"Var4","KernelFunction","gaussian");
[~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
plotSVM(svmdataD_test,scores,x1Grid,x2Grid,'Gaussian kernel');
%% 
% Задание 4e

clear
svmdataE = readtable('svmdata_e.txt')
svmdataE = svmdataE(:,2:end);
svmdataE_test = readtable('svmdata_e_test.txt')
svmdataE_test = svmdataE_test(:,2:end);

scaleOptions = {'auto' 1 10 100};
[x1Grid,x2Grid] = meshgrid(linspace(min(svmdataE_test{:,1}),max(svmdataE_test{:,1}),100),...
    linspace(min(svmdataE_test{:,2}),max(svmdataE_test{:,2}),100));
for scale = scaleOptions
    svmModel = fitcsvm(svmdataE,"Var4","KernelFunction","linear","KernelScale",scale{1});
    [~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
    plotSVM(svmdataE_test,scores,x1Grid,x2Grid,'scale = ' + string(scale{1}) +', linear kernel');
    svmModel = fitcsvm(svmdataE,"Var4","KernelFunction","polynomial","PolynomialOrder",1,"KernelScale",scale{1});
    [~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
    plotSVM(svmdataE_test,scores,x1Grid,x2Grid,'scale = ' + string(scale{1}) + ', polynomial kernel, order = 1');
    svmModel = fitcsvm(svmdataE,"Var4","KernelFunction","polynomial","PolynomialOrder",2,"KernelScale",scale{1});
    [~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
    plotSVM(svmdataE_test,scores,x1Grid,x2Grid,'scale = ' + string(scale{1}) + ', polynomial kernel, order = 2');
    svmModel = fitcsvm(svmdataE,"Var4","KernelFunction","polynomial","PolynomialOrder",3,"KernelScale",scale{1});
    [~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
    plotSVM(svmdataE_test,scores,x1Grid,x2Grid,'scale = ' + string(scale{1}) + ', polynomial kernel, order = 3');
    svmModel = fitcsvm(svmdataE,"Var4","KernelFunction","polynomial","PolynomialOrder",4,"KernelScale",scale{1});
    [~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
    plotSVM(svmdataE_test,scores,x1Grid,x2Grid,'scale = ' + string(scale{1}) + ', polynomial kernel, order = 4');
    svmModel = fitcsvm(svmdataE,"Var4","KernelFunction","polynomial","PolynomialOrder",5,"KernelScale",scale{1});
    [~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
    plotSVM(svmdataE_test,scores,x1Grid,x2Grid,'scale = ' + string(scale{1}) + ', polynomial kernel, order = 5');
    svmModel = fitcsvm(svmdataE,"Var4","KernelFunction","gaussian");
    [~,scores] = predict(svmModel,[x1Grid(:),x2Grid(:)]);
    plotSVM(svmdataE_test,scores,x1Grid,x2Grid,'scale = ' + string(scale{1}) + ', Gaussian kernel');
end
%% 
% Задание 5a

clear
glass = readtable('glass.csv');
glass = glass(:,2:end)
cvpt = cvpartition(glass.Type,"HoldOut",0.2);
dataTrain = glass(training(cvpt),:);
dataTest = glass(test(cvpt),:);
%%
treeModel = fitctree(dataTrain,"Type")
view(treeModel,'Mode','graph')
prediction = predict(treeModel,dataTest);
accuracy = nnz(prediction == dataTest{:,end})/numel(prediction)
%%
maxSplitsOptions = 1:10;
minLeafOptions = 1:5;
pruneOptions = ["on" "off"];
pruneCriterionOptions = ["error" "impurity"];
mergeOptions = ["on" "off"];
optionTable = table('Size',[numel(maxSplitsOptions)*numel(minLeafOptions)*numel(pruneOptions)*numel(pruneCriterionOptions)*numel(mergeOptions) 6],'VariableTypes',{'int8','int8','string','string','string','double'},'VariableNames',{'Max Splits','Min Leaf Size','Prune','Prune Criterion','Merge Leaves','Accuracy'});
tIndex = 1;
for maxSplits = maxSplitsOptions   
    for minLeaf = minLeafOptions
        for isPrune = pruneOptions
            for pruneCrit = pruneCriterionOptions
                for mergeLeaves = mergeOptions
                    optionTable{tIndex,1} = maxSplits;
                    optionTable{tIndex,2} = minLeaf;
                    optionTable{tIndex,3} = isPrune;
                    optionTable{tIndex,4} = pruneCrit;
                    optionTable{tIndex,5} = mergeLeaves;
                    treeModel = fitctree(glass,"Type","MaxNumSplits",maxSplits,"MinLeafSize",minLeaf,"Prune",isPrune,"PruneCriterion",pruneCrit,"MergeLeaves",mergeLeaves);
                    prediction = predict(treeModel,dataTest);
                    optionTable{tIndex,6} = nnz(prediction == dataTest{:,end})/numel(prediction);
                    tIndex = tIndex + 1;
                end
            end
        end
    end
end
optionTable
%%
optionTable = sortrows(optionTable,'Accuracy','descend')
optionTable(1:10,:)
%% 
% Задание 5b

clear
spam7 = readtable('spam7.csv')
cvpt = cvpartition(spam7.yesno,"HoldOut",0.2);
dataTrain = spam7(training(cvpt),:);
dataTest = spam7(test(cvpt),:);
treeModel = fitctree(dataTrain,"yesno","MaxNumSplits",10)
view(treeModel,'Mode','graph')
prediction = predict(treeModel,dataTest);
accuracy = nnz(string(prediction) == string(dataTest{:,end}))/numel(prediction)
confusionchart(dataTest{:,end},prediction)
%% 
% Задание 6

clear
dataTrain = readtable('bank_scoring_train.csv')
dataTest = readtable('bank_scoring_test.csv')

knnModel = fitcknn(dataTrain,"SeriousDlqin2yrs","NumNeighbors",5);
prediction = predict(knnModel,dataTest);
accuracy = nnz(prediction == dataTest{:,1})/numel(prediction)
confusionchart(dataTest{:,1},prediction)
title('KNN classifier')

treeModel = fitctree(dataTrain,"SeriousDlqin2yrs","MaxNumSplits",10);
prediction = predict(treeModel,dataTest);
accuracy = nnz(prediction == dataTest{:,1})/numel(prediction)
confusionchart(dataTest{:,1},prediction)
title('Tree classifier')
view(treeModel,'Mode','graph')
%%
function plotDependence(data, options)    
    partRatios = 0.05:0.05:0.95;
    accuracy = zeros(1,numel(partRatios));
    accIndex = 1;
    for ratio = partRatios
        cvpt = cvpartition(data{:,end},"HoldOut",ratio);
        dataTrain = data(training(cvpt),:);
        dataTest = data(test(cvpt),:);
        nbModel = fitcnb(dataTrain,dataTrain(:,end),"DistributionNames", options);
        prediction = predict(nbModel,dataTest);
        accuracy(accIndex) = nnz(string(prediction) == string(dataTest{:,end}))/numel(prediction);
        accIndex = accIndex + 1;
    end
    figure
    plot(partRatios,accuracy,'o-')
    xlabel("Test data ratio");
    ylabel("Accuracy");
    grid on
end

function plotSVM(testData, scores, x1Grid, x2Grid, plotTitle)
    figure;
    fill([min(testData{:,1}) min(testData{:,1}) max(testData{:,1}) max(testData{:,1})],[min(testData{:,2}) max(testData{:,2}) max(testData{:,2}) min(testData{:,2})],'g')
    hold on
    map = [1 0.3 0.3];
    M = contourf(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
    M = M(:, 2:end);
    colormap(map)
    h = zeros(2,1);
    firstEl = testData{1,end};
    if string(firstEl{1}) == "red"
        h(1:2) = gscatter(testData{:,1},testData{:,2},testData{:,end},[0.7 0 0; 0 0.7 0],'ox');
        legend(h,{'red','green'},'Location','Southeast');
    else
        h(1:2) = gscatter(testData{:,1},testData{:,2},testData{:,end},[0 0.7 0; 0.7 0 0],'xo');
        legend(h,{'green','red'},'Location','Southeast');
    end    
    title(plotTitle)    
    hold off
end