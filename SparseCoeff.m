function [Coeff_Train Coeff_Test] = SparseCoeff( fea_Train , fea_Test )

% find a sparse representation coefficient matrix such that fea_Train=fea_Train*Coeff_Train, and fea_Test=fea_Train*Coeff_Test.

% Input:
% fea_Train dim * num_Train, training data matrix, each column is a sample
% fea_Test  dim * num_Test, test data matrix, each column is a sample

% Output:
% Coeff_Train num_Train*num_Train
% Coeff_Test  num_Train*num_Test

[dim,num_Train] = size( fea_Train ) ;

% normalization 
for i = 1 : ntrn
    fea_Train(:,i) = fea_Train(:,i) / norm( fea_Train(:,i) ) ;
end
for i = 1 : ntst
    fea_Test(:,i) = fea_Test(:,i) / norm( fea_Test(:,i) ) ;
end



% SPAMS package
param.lambda = 0.05 ;        % not more than 20 non-zeros coefficients
%     param.numThreads=2;        % number of processors/cores to use; the default choice is -1
%     and uses all the cores of the machine
param.mode = 1 ;             % penalized formulation
param.verbose = false ;      % no output
    

% compute Coeff_Train
Coeff_Train = zeros( num_Train , num_Train ) ;
for i = 1 : num_Train
    y = fea_Train(:,i) ;
    if i == 1
        D = fea_Train(:,2:num_Train) ;
    elseif i == num_Train
        D = fea_Train(:,1:num_Train-1) ;
    else
        D = fea_Train(:,[1:i-1,i+1:num_Train]) ;
    end
    wi = mexLasso( y , D , param ) ;    
    if i == 1
        Coeff_Train(2:num_Train,i) = wi(1:num_Train-1) ;
    elseif i == num_Train
        Coeff_Train(1:num_Train-1,i) = wi(1:num_Train-1) ;
    else
        Coeff_Train([1:i-1,i+1:num_Train],i) = wi(1:num_Train-1) ;
    end
%     fprintf('No.%d sample is done\n',i)
end


if nargin < 2
    Coeff_Test = [] ;
else
    % compute Coeff_Test
    Coeff_Test = mexLasso( fea_Test , fea_Train , param ) ;
end


