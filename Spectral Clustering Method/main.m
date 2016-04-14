load('train.csv')
load('test.csv')
user_movie_whole = train;
%% Cross Validation 
Kclu = [10 30 50 100 200 500]';
knearst = [50 200 1000]';
redim = [2 100 500]';

% Cross Validation for normalized cosine by different K clusters
for i=1:size(Kclu,1)
    [mse, correct] = Cv_Spec_Cluster(user_movie_whole, 5, 'normalized', 'NA', 'NA', 'cosine', 'NA', Kclu(i));
    disp(mse)
    disp(correct)
end

% Cross Validation for normalized knn-cosine by different K clusters and k-nearst neigborhood
for i=1:size(Kclu,1)
    for j=1:size(knearst,1)
        [mse, correct] = Cv_Spec_Cluster(user_movie_whole, 5, 'normalized', 'NA', 'NA', 'knn_cosine', knearst(j), Kclu(i));
        disp(mse)
        disp(correct)
    end
end

% Cross Validation for unnormalized cosine by different K clusters
for i=1:size(Kclu,1)
    [mse, correct] = Cv_Spec_Cluster(user_movie_whole, 5, 'normalized', 'NA', 'NA', 'cosine', 'NA', Kclu(i));
    disp(mse)
    disp(correct)
end

% Cross Validation for unnormalized cosine by different K clusters and knearst KNN

for i=1:size(Kclu,1)
    for j=1:size(knearst,1)
        [mse, correct] = Cv_Spec_Cluster(user_movie_whole, 5, 'unnormalized', 'NA', 'NA', 'knn_cosine', knearst(j), Kclu(i));
        disp(mse)
        disp(correct)
    end
end

% CV for normalized PCA cosine by different K clusters and reduced dimensions
cvmse5 = zeros(size(Kclu,1),3);
cvcorrect5 = zeros(size(Kclu,1),3);
for i=1:size(Kclu,1)
    for j=1:size(redim,1)
        [mse, correct] = Cv_Spec_Cluster(user_movie_whole, 5, 'normalized', 'pca', redim(j), 'cosine', 'NA', Kclu(i));
        disp(mse)
        disp(correct)
    end
end


% CV for normalized PCA cosine by different K clusters and reduced dimensions
for i=1:size(Kclu,1)
    for j=1:size(redim,1)
        for p = 1:size(knearst,1)
            [mse, correct] = Cv_Spec_Cluster(user_movie_whole, 5, 'normalized', 'pca', redim(j), 'knn_cosine', knearst(p), Kclu(i));
            disp(mse)
            disp(correct)
        end
    end
end

%% Prediction
predicts = Pred_Spec_Cluster(user_movie_whole, test, 'normalized', 'NA', 'NA', 'cosine', 'NA', 30);
