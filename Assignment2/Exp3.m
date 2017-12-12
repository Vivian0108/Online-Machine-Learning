clear;

K = 10;
T = 1E+04;
delta = 0.1;
eta = sqrt(2*log(K)/(T*K));

w_unif = 0.001*eta;

p = ones(1, K);
agg_cost = zeros(1, K);
pseudo_regret = 0;

figure(1);

for t=1:T
    Z = sum(p);
    p = p/Z;
    
    cdf = cumsum(p);
    
    X = rand;
    I = 1;
    
    for j=1:K
        if(X <= cdf(j))
            I = j;
            break;
        end
    end

    V = zeros(1, K);

    if(I <= K-2)
        V(I) = binornd(1, 0.5)/p(I);
    end
    if(I == K-1)
        V(I) = binornd(1, 0.5 - delta)/p(I);
    end
    if(I == K)
        if(t <= T/2)
            V(I) = binornd(1, 0.5 + delta)/p(I);
        end
        if(t > T/2)
            V(I) = binornd(1, 0.5 - 2*delta)/p(I);
        end
    end
    
    agg_cost = agg_cost + V;
    
    V_exp = exp(-agg_cost*eta);
    
    for j=1:K
        p(j) = (1 - w_unif)*V_exp(j)/sum(V_exp) + w_unif/K;
    end

    pseudo_regret = max(agg_cost(I) - min(agg_cost), pseudo_regret);
    
    hold on;
    plot(t, pseudo_regret, 'r*');

end