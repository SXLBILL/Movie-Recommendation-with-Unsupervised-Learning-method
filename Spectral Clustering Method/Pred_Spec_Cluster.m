function [predicts, cl, eigVec, eigVal] = Pred_Spec_Cluster(user_movie_whole, testdata, sparsity, ispca, redim, distancef, knearst, Kclu)
    %% Input
    % user_movie_whole is a [movie_id, customer_id,rating] matrix
    % testdata is data predicting
    % sparsity is normalized or unnormalizd option
    % ispca is PCA option
    % redim is the demension choosed by PCA
    % distancef is the distance funtion choosed such as knn-cosine
    % knearst is choosed nearst neighboorhood for KNN
    % Kclu is the cluster numbers
    %% Output
    % predicts is the predicted value
    % cl is the cluster label each customer from training set belong to
    % eigVec is the eigen vector from Lv = lambda*Dv
    % eigVal is the eigen value from Lv = lambda*Dv
    
    %% load trainind data and test data
    XY_test = testdata;
    XY_train = user_movie_whole;
    NumCst = length(unique(XY_train(:,2)));
    NumMov = length(unique(XY_train(:,1)));
    % For XY_train; col is [ori mov_id, ori cust_id, rating]
    % Match ori mov_id with new mov_id and ori cust_id with new cust_id
    cstMatch = [unique(XY_train(:,2)) [1:NumCst]'];
    movMatch = [unique(XY_train(:,1)) [1:NumMov]'];
    % Match XY_train with new cust_id and new mov_id
    [~,cst2] = ismember(XY_train(:,2), cstMatch(:,1));
    [~,mov2] = ismember(XY_train(:,1), movMatch(:,1));
    % For XY_train2; col is [ori cust_id, new cust_id, old mov_id, new
    % mov_id, rating]
    XY_train2 = [XY_train(:,2) cst2 XY_train(:,1) mov2 XY_train(:,3)];
    %% Calculate UMR
    UMR = zeros(NumCst, NumMov);

    % user2, movie2 and rating
    for i=1:size(XY_train2, 1)
        UMR(XY_train2(i,2), XY_train2(i,4)) = XY_train2(i,5);
    end
    
    % normalized UMR
    if(strcmp(sparsity,'normalized'))
        for i=1:size(UMR,2)
            UMR(UMR(:,i)==0,i) = mean(UMR(UMR(:,i)~=0, i));
        end
        UMR_end = UMR;
        % PCA normalized UMR
        if(strcmp(ispca, 'pca'))  
            [UMR_PCA, score, latent] = pca(UMR');
            UMR = UMR_PCA(:,1:redim);
        end
        % cosine W
        W = pdist(UMR, 'cosine');
        W = squareform(W, 'tomatrix');
        W = 1 - W;
        W(W<0)=0;
    end
    
    % unnormalized UMR
    if(strcmp(sparsity,'unnormalized'))
        UMR(UMR==0)=NaN;
        UMR_end = UMR;
        % cosine W
        W = zeros(size(UMR,1), size(UMR,1));
        for i=1:size(UMR,1)
            disp(i)
            for j=i:size(UMR,1)
                UnaIdx = intersect(find(~isnan(UMR(i,:))), find(~isnan(UMR(j,:))));
                W(i,j) = UMR(i,UnaIdx)*UMR(j,UnaIdx)'/(sqrt(UMR(i, UnaIdx)*UMR(i, UnaIdx)')*sqrt(UMR(j, UnaIdx)*UMR(j, UnaIdx)'));
            end
        end
        for j=1:size(UMR,2)
            for i=(j+1):size(UMR,1)
                W(i,j) = W(j,i);
            end
        end
        W(isnan(W))=0;
    end

    %% KNN_cosine
    if(strcmp(distancef,'knn_cosine'))
        % KNN to find neiborhood
        for i = 1:size(W,1)
            W(i,i) = -1;
        end
        KNN = zeros(size(W,1), size(W,1));
        for i=1:size(W,1)
            [z, idx] = sort(W(i,:), 2, 'descend');
            KNN(i, idx(1:knearst))=1;
            KNN(idx(1:knearst), i)=1;
        end
        % two nodes must be connected
        Wknn = KNN.*(KNN==KNN');
        W = Wknn;
    end
    %% spectral cluster and calculate mse
    [mse, predicts, MSEs, cl, eigVec, eigVal] = spectral_clustering(UMR_end, W, cstMatch, movMatch, XY_test, Kclu);
end