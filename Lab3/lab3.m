%% 
% Задание 1

clear
format long
data = readtable("pluton.csv")
data_real = table2array(data);
criterionValues = zeros(20,2);
for numIterations = 1:20
    clusters = kmeans(data_real,3,'MaxIter',numIterations);
    figure
    silhouette(data_real,clusters)
    title("numIterations = " + string(numIterations))
    evaluation = evalclusters(data_real,clusters,'silhouette');
    criterionValues(numIterations,1) = evaluation.CriterionValues;
end
data_norm = normalize(data_real,1);
for numIterations = 1:20
    clusters = kmeans(data_norm,3,'MaxIter',numIterations);
    figure
    silhouette(data_norm,clusters)
    title("numIterations = " + string(numIterations))
    evaluation = evalclusters(data_norm,clusters,'silhouette');
    criterionValues(numIterations,2) = evaluation.CriterionValues;
end
criterionValues
%% 
% Задание 2

clear
data_1 = table2array(readtable("clustering_1.csv"))
data_2 = table2array(readtable("clustering_2.csv"))
data_3 = table2array(readtable("clustering_3.csv"))

evaluation_1 = evalclusters(data_1,'kmeans','silhouette','KList',1:10);
clusters = kmeans(data_1,evaluation_1.OptimalK);
figure
gscatter(data_1(:,1),data_1(:,2),clusters)
title('K-means, dataset 1')
evaluation_2 = evalclusters(data_2,'kmeans','silhouette','KList',1:10);
clusters = kmeans(data_2,evaluation_2.OptimalK);
figure
gscatter(data_2(:,1),data_2(:,2),clusters)
title('K-means, dataset 2')
evaluation_3 = evalclusters(data_3,'kmeans','silhouette','KList',1:10);
clusters = kmeans(data_3,evaluation_3.OptimalK);
figure
gscatter(data_3(:,1),data_3(:,2),clusters)
title('K-means, dataset 3')

evaluation_1 = evalclusters(data_1,'linkage','silhouette','KList',1:10);
tree = linkage(data_1);
clusters = cluster(tree,'maxclust',evaluation_1.OptimalK);
figure
gscatter(data_1(:,1),data_1(:,2),clusters)
title('Linkage, dataset 1')
evaluation_2 = evalclusters(data_2,'linkage','silhouette','KList',1:10);
tree = linkage(data_2);
clusters = cluster(tree,'maxclust',evaluation_2.OptimalK);
figure
gscatter(data_2(:,1),data_2(:,2),clusters)
title('Linkage, dataset 2')
evaluation_3 = evalclusters(data_3,'linkage','silhouette','KList',1:10);
tree = linkage(data_3);
clusters = cluster(tree,'maxclust',evaluation_3.OptimalK);
figure
gscatter(data_3(:,1),data_3(:,2),clusters)
title('Linkage, dataset 3')

clusters = dbscan(data_1,0.5,5);
evaluation_1 = evalclusters(data_1,clusters,'silhouette');
figure
gscatter(data_1(:,1),data_1(:,2),clusters)
title('DBSCAN, dataset 1')
clusters = dbscan(data_2,0.5,5);
evaluation_2 = evalclusters(data_2,clusters,'silhouette');
figure
gscatter(data_2(:,1),data_2(:,2),clusters)
title('DBSCAN, dataset 2')
clusters = dbscan(data_3,0.5,5);
evaluation_3 = evalclusters(data_3,clusters,'silhouette');
figure
gscatter(data_3(:,1),data_3(:,2),clusters)
title('DBSCAN, dataset 3')
%% 
% Задание 3

clear
test_image = imread('Кошка 1440x900.jpg');
imshow(test_image)
image_features = double(reshape(test_image,[],3));
numColors = 15;
[clusters,centers] = kmeans(image_features,numColors,'MaxIter',500);
features_compressed = zeros(size(image_features));
for i = 1:size(clusters,1)
    features_compressed(i,:) = centers(clusters(i),:);
end
image_compressed = uint8(reshape(features_compressed,900,1440,3));
imshow(image_compressed)
%% 
% Задание 4

clear
data = readtable('votes.csv');
data.X1856 = str2double(data.X1856);
data.X1860 = str2double(data.X1860)
data_real = table2array(data);
data_real = data_real(all(~isnan(data_real),2),:)
clusters = linkage(data_real);
dendrogram(clusters)