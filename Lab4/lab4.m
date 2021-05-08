%% 
% Задание 1

clear
data = readtable("glass.csv");
data = data(:,2:end)
maxNumCycles = 100;
bagLoss = zeros(maxNumCycles,1);
for numCycles = 1:maxNumCycles
    bagModel = fitcensemble(data,"Type","Method","Bag","Learners","tree","NumLearningCycles",numCycles,"KFold",5);
    bagLoss(numCycles) = kfoldLoss(bagModel);
end
plot(bagLoss)
%%
for numCycles = 1:maxNumCycles
    bagModel = fitcensemble(data,"Type","Method","Bag","Learners","discriminant","NumLearningCycles",numCycles,"KFold",5);
    bagLoss(numCycles) = kfoldLoss(bagModel);
end
plot(bagLoss)
%% 
% Задание 2

clear
data = readtable("vehicle.csv");
data = data(:,2:end)
%%
maxNumCycles = 100;
boostLoss = zeros(maxNumCycles,1);
for numCycles = 1:maxNumCycles
    boostModel = fitcensemble(data,"Class","Method","AdaBoostM2","Learners","tree","NumLearningCycles",numCycles,"KFold",5);
    boostLoss(numCycles) = kfoldLoss(boostModel);
end
plot(boostLoss)
%%
for numCycles = 1:maxNumCycles
    boostModel = fitcensemble(data,"Class","Method","AdaBoostM2","Learners","discriminant","NumLearningCycles",numCycles,"KFold",5);
    boostLoss(numCycles) = kfoldLoss(boostModel);
end
plot(boostLoss)
%% 
% Задание 3

clear
data = readtable("titanic_train.csv");
data.PassengerId = [];
data.Name = [];
data.Ticket =[];
data.Age(isnan(data.Age)) = mean(data.Age(~isnan(data.Age)))
models{1} = fitcsvm(data,"Survived","KernelFunction","linear");
models{2} = fitcsvm(data,"Survived","KernelFunction","gaussian");
models{3} = fitctree(data,"Survived");
models{4} = fitcensemble(data,"Survived","Method","Bag");

modelLoss = zeros(4,1);
cvpt = cvpartition(data.Survived,"KFold",5);
for i = 1:4
    cvModel = crossval(models{i});
    modelLoss(i) = kfoldLoss(cvModel);    
end
plot(modelLoss)