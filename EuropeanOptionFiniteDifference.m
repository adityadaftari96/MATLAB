% Finite Difference for European Call Option

T = 1;
r = 0.05;
sigma = 0.3;
K = 100;
R = 6;
N = 2; M = 4;

trials = 5;
u_mat = NaN(M*2^(trials+1)+1, N*4^trials+1);
ttm_arr = NaN(N*4^trials+1, 1);

for x = 1:trials
    N = N*4;
    M = M*2;
    dt = T/N;
    dy = R/M;
    % N*dt = T; M*dy = R
    a = (-0.5*dt*(sigma^2)/(dy^2)) + (0.5*dt*(r-0.5*sigma^2)/dy);
    b = 1 + dt*(r + (sigma^2/(dy^2)));
    c = dt*(-0.5*((r - 0.5*(sigma^2))/dy) - 0.5*(sigma^2)/(dy^2));
    temp = reshape([ones(2*M -1, 1) *a; ones(2*M -1, 1) *b; ones(2*M -1, 1) *c], 2*M - 1, 3);
    Ah = spdiags(temp, [-1, 0, 1], 2*M - 1, 2*M - 1);
    % u at state i and time n=0
    px = exp((-M:M)' * dy);
    uin = max(px-K, 0);
    if x == trials
        u_mat(:, 1) = uin;
        ttm_arr(1) = 0;
    end
    for n = 1:N
        Bhn = uin(2:2*M);
        Bhn(end) = Bhn(end) -c*( exp(M*dy) - exp(-r*(n+1)*dt)*K );
        uinp1 = Ah\Bhn;
        uin = [0; uinp1; exp(M*dy) - exp(-r*(n+1)*dt)*K];

        if x == trials
            u_mat(:, n+1) = uin;
            ttm_arr(n+1) = n*dt;
        end
    end
end


mesh(ttm_arr, px, u_mat)


% hold off
% legend show

figure()
plot(px, uin)
title('European Call Option Price vs Spot Price')
ylabel({'Call Option Price'});
xlabel({'Spot Price'});
