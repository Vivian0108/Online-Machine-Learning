clear;
clc;

d = 10;
T = 1E+03;
delta = 0.1;
c = linspace(0.1,2.1,11);
eta = c*sqrt(2*log(d)/T);
pseudo_regret = zeros(1,11);
count = 0;


for path_count = 1:50    

    for eta_count = 1:11
        
        w = ones(1, d);        
        
        for t=1:T
            count = count+1;
            zt = sum(w);
            w = w/zt;
            I = 1;
            cumul_w = cumsum(w);    
            X = rand;    
    
            for i=1:d
                if(X <= cumul_w(i))
                    I = i;
                    break;
                end
            end
    
            v = zeros(1,d);
    
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
                        v(j) = binornd(1, 0.5 - 2*delta);
                    end
                end
            end
        
            cost = w*v';
        
            for j=1:d
                w(j) = w(j)*exp(-eta(eta_count)*v(j));
            end
        
            v1 = min(v);
            pseudo_regret(eta_count) = pseudo_regret(eta_count) + cost - v1;        
            fprintf('count = %d\n',count);
        end
    end
end
figure(1);
pseudo_regret = pseudo_regret/50;
plot(eta,pseudo_regret,'b-*');
