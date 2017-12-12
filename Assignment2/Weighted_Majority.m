clear;

d = 10;
T = 1E+04;
delta = 0.1;
eta = sqrt(2*log(d)/T);

weight = ones(1, d);
agg_cost = zeros(1, d);
pseudo_regret = 0;

figure(1);

for t=1:T
    Z = sum(weight);
    weight = weight/Z;
    
    cdf = cumsum(weight);
    
    X = rand;
    I = 1;
    
    for j=1:d
        if(X <= cdf(j))
            I = j;
            break;
        end
    end
    
    cost = zeros(1, d);
    
    for j=1:d
        if(j <= d-2)
            cost(j) = binornd(1, 0.5);
        end
        if(j == d-1)
            cost(j) = binornd(1, 0.5 - delta);
        end
        if(j == d)
            if(t <= T/2)
                cost(j) = binornd(1, 0.5 + delta);
            end
            if(t > T/2)
                cost(j) = binornd(1, 0.5 - 2*delta);
            end
        end
    end
    
    Incurred_cost = weight*cost';
    
    for j=1:d
        weight(j) = weight(j)*exp(-eta*cost(j));
    end
    
    agg_cost = agg_cost + cost;
    
    pseudo_regret = max(agg_cost(I) - min(agg_cost), pseudo_regret);
    
    hold on;
    plot(t, pseudo_regret, 'r*');
end