clear;
clc;

K = 10;
T = 1E+03;
delta = 0.1;
c=linspace(0.1,2.1,11);
eta = c*sqrt(2*log(K)/(T*K));

pseudo_regret = zeros(1,11);
count = 0;


for round=1:50
    for eta_count = 1:11
        w_unif = 0.001*eta(eta_count);
        p = ones(1, K);
        L = zeros(1, K);
        for t=1:T
            count = count +1;
            Z = sum(p);
            p = p/Z;
    
            cumul_p = cumsum(p);
    
            X = rand;
            I = 1;
    
            for i=1:K
                if(X <= cumul_p(i))
                    I = i;
                    break;
                end
            end
    
            v = zeros(1, K);
        
            if(I <= K-2)
                v(I) = binornd(1, 0.5)/p(I);
            end
            if(I == K-1)
                v(I) = binornd(1, 0.5 - delta)/p(I);
            end
            if(I == K)
                if(t <= T/2)
                    v(I) = binornd(1, 0.5 + delta)/p(I);
                end
                if(t > T/2)
                    v(I) = binornd(1, 0.5 - 2*delta)/p(I);
                end
            end
    
            L = L + v;
        
            L_weights = exp(-L*eta(eta_count));
    
            for j=1:K
                p(j) = L_weights(j)/sum(L_weights);
            end

            pseudo_regret(eta_count) = pseudo_regret(eta_count) + L(I) - min(L);
            fprintf('count = %d\n',count);
                    
        end
    end
end
figure(1);
 pseudo_regret =  pseudo_regret/50;
 plot(eta, pseudo_regret,'b--');