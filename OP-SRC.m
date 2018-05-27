function [eigvector eigvalue] = OP_SRC( X , Y )

% reference:
% Optimized Projections for Sparse Representation based Classification 
% Can-Yi Lu, De-Shuang Huang
% Neurocomputing 113: 213-219 (2013)

% X(d*n) , Y(1*n) each column is a sample

% pca projection first

[ eigvector_PCA , eigvalue ] = PCA( fea_Train ) ;
X = eigvector_PCA' * X ;

[d,n] = size(X) ;
Rw = zeros(d,d) ;
Rb = zeros(d,d) ;
nclass = length( unique(Y) ) ;
r = zeros(d,nclass) ;
epsilon = 10^-1 ;
Jp_old = -1 ;
corr=0;
loop = 0 ;
Xk = X ;
while 1
    loop = loop + 1 ;
    if loop == 2
        break ;
    end         
    A = SparseCoeff( Xk ) ;
    for i = 1 : n
        x = X(:,i) ;                              
        residual = zeros(nclass,1) ;
        for k = 1 : nclass
            ind = find(Y~=k) ;
            a = A(:,i) ;                          
            a(ind) = 0 ;  
            r(:,k) = x - X * a ;     
            residual(k) = norm(r(:,k)) ;
        end
        ri = r(:,Y(i)) * r(:,Y(i))' ;
        Rw = Rw + ri ;
        Rb = Rb + r * r' - ri ;        
        [minres,kth] = min(residual) ;
%         residual
        if(kth==Y(i))
            corr = corr+1 ;
        end       
    end
    corRate = corr / n ;    
    Rw = Rw/n ;
    Rb = Rb/(n*(nclass-1)) ;
    beta = 0.3 ;
    [eigvector eigvalue] = eig( beta * Rb - Rw ) ;    % our method
    eigvalue = diag(eigvalue) ;
    [junk, index] = sort(-eigvalue) ;
    eigvalue = eigvalue(index) ;
    eigvector = eigvector(:,index) ;   
end

for tmp = 1:size(eigvector,2)
    eigvector(:,tmp) = eigvector(:,tmp)./norm(eigvector(:,tmp));
end

eigvector = eigvector_PCA * eigvector ;

