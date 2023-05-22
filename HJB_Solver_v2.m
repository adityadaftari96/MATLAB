% Finite Difference for optimal trend following trading rules

% % Case a
% lambda1 = 0.2;
% lambda2 = 30;
% mu1 = 0.15;
% mu2 = 0.1;
% sigma = 0.2;
% Kb = 0.0006;
% Ks = 0.0006;
% r = 0.085;
% T = 1;

% % Case b
% lambda1 = 20;
% lambda2 = 1;
% mu1 = 0.2;
% mu2 = 0.0;
% sigma = 0.45;
% Kb = 0.05;
% Ks = 0.05;
% r = 0.08;
% T = 1;

% % Market Parameters
lambda1 = 0.36;
lambda2 = 2.53;
mu1 = 0.18;
mu2 = -0.77;
sigma = 0.184;
Kb = 0.001;
Ks = 0.001;
r = 0.0679;
T = 1;

% % Grid Parameters
R = 1;
N = 600;
M = N;

ttm_arr = NaN(N+1, 1);
pb_boundary_arr = NaN(N+1, 1);
ps_boundary_arr = NaN(N+1, 1);

ttm_arr(1) = 0;
pb_boundary_arr(1) = 1;
% ps_boundary_arr(1) = 1;
ps_boundary_arr(1) = (r - mu2 + 0.5*(sigma^2))/(mu1-mu2);


dt = T/N;
dp = R/M;
% N*dt = T; M*dp = R
i_rng = (1:M-1)';

factor1 = dt*((mu1-mu2)*i_rng.*(1-i_rng*dp)/sigma).^2;
factor2 = (dt/dp)*((lambda1+lambda2)*i_rng*dp - lambda2);

a = -0.5*factor1;
b = 1 + factor1 - factor2;
c = -0.5*factor1 + factor2;

p = (1:M-1)' * dp;
f_p = (mu1-mu2)*p + mu2 - 0.5*sigma^2;

% u at state i and n=0. (or Time = T - n*dt = T)
u0_i_n = zeros(M-1, 1);
u1_i_n = ones(M-1, 1)*log(1-Ks);


for n = 1:N
    u0_0 = n*dt*(r + lambda2);
    u1_0 = n*dt*(f_p(1) + lambda2) + log(1-Ks);

    u0_end = n*dt*(r - lambda1);
    u1_end = n*dt*(f_p(end) - lambda1) + log(1-Ks);

    B0_h_n = u0_i_n(2:end-1) + r*dt;
    B1_h_n = u1_i_n(2:end-1) + f_p(2:end-1)*dt;

    B0_h_n(1) = B0_h_n(1) - a(1)*u0_0;
    B0_h_n(end) = B0_h_n(end) - c(end)*u0_end;

    B1_h_n(1) = B1_h_n(1) - a(1)*u1_0;
    B1_h_n(end) = B1_h_n(end) - c(end)*u1_end;

    u0_i_n_ = TDMAsolver(a,b,c,B0_h_n)';
    u1_i_n_ = TDMAsolver(a,b,c,B1_h_n)';

    d0 = (u0_i_n_ - u1_i_n_ + log(1+Kb)) > 0;

    u0_1 = d0.*u0_i_n_;
    u0_2 = not(d0) .* (u1_i_n_ - log(1+Kb));
    u0_i_n_temp = u0_1 + u0_2;
    u0_i_n = [u0_0; u0_i_n_temp; u0_end];

    d1 = (u1_i_n_ - u0_i_n_ - log(1-Ks)) > 0;

    u1_1 = d1.*u1_i_n_;
    u1_2 = not(d1) .* (u0_i_n_ + log(1-Ks));
    u1_i_n_temp = u1_1 + u1_2;
    u1_i_n = [u1_0; u1_i_n_temp; u1_end];

    ttm_arr(n+1) = n*dt;
    if sum(not(d0)) == 0
        pb_boundary_arr(n+1) = 1;
    else
        pb_boundary_arr(n+1) = p(find(not(d0), 1, 'first'));
    end

    if sum(d1) == 0
        ps_boundary_arr(n+1) = 0;
    else
        ps_boundary_arr(n+1) = p(find(d1, 1, 'first'));
    end
end

time_arr = flip(ttm_arr);

figure()
plot(time_arr, pb_boundary_arr, time_arr, ps_boundary_arr)
% ylim([0.7 1.0])
title('Optimal Buy and Sell Boundaries')
ylabel({'p'});
xlabel({'t'});
