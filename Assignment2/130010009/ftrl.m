clear;
clc;

d = 10;
T = 1E+03;
delta = 0.1;
c = linspace(0.1,2.1,11);
eta = c*sqrt(log(d)/(2*T));

pseudo_regret = zeros(1,11);
count = 0;


for path_count = 1:50    

    for eta_count = 1:11        

        w = ones(1, d);
        L = zeros(1, d);
        for t=1:T
            Z = sum(w);
            w = w/Z;
            count = count+1;
        
            cumul_w = cumsum(w);
        
            X = rand;
            I = 1;
        
            for j=1:d
                if(X <= cumul_w(j))
                    I = j;
                    break;
                end
            end
            
            v = zeros(1, d);
        
            for j=1:d
                if(j <= d-2)
                    v(j) = binornd(1, 0.5);
                end
                if(j == d-1)
                    v(j) = binornd(1, 0.5 - delta);
                end
                if(j == d)
                    if(t <= T/2)
                        v(j) = binornd(1, 0.5 + delta);
                    end
                    if(t > T/2)
                        v(j) = binornd(1, 0.5 + 2*delta);
                    end
                end
            end
    
            cost = w*v';
        
            L = L + v;
        
            for j=1:d
                w(j) = exp(-eta(eta_count)*L(j));
            end
            v1 = min(v);
            pseudo_regret(eta_count) = pseudo_regret(eta_count) + cost - v1;
            fprintf('count = %d\n',count);
        end
    end
end
figure(1);
pseudo_regret = pseudo_regret/50;
plot(eta,pseudo_regret,'r-*');