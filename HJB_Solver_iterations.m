% Finite Difference for optimal trend following trading rules

% Market Parameters
lambda1 = 0.36;
lambda2 = 2.53;
mu1 = 0.18;
mu2 = -0.77;
sigma = 0.184;
Kb = 0.001;
Ks = 0.001;
r = 0.0679;
T = 1;

% Grid Parameters
R = 1;
N = 4; M = 4;

trials =7;
err_roc = NaN(trials,2);
row_names = cell(1, trials);

% u_mat = NaN(M*2^trials+1, N*2^trials+1);
ttm_arr = NaN(N*2^trials+1, 1);
pb_boundary_arr = NaN(N*2^trials+1, 1);
ps_boundary_arr = NaN(N*2^trials+1, 1);

for x = 1:trials
    % if x > 1
    %     uin_prev = uin;
    %     px_prev = p;
    % end
    N = N*2;
    M = M*2;
    row_names(x) = compose('%dx%d',N,M);

    dt = T/N;
    dp = R/M;
    % N*dt = T; M*dp = R
    i_rng = (1:M-1)';

    factor1 = dt*((mu1-mu2)*i_rng.*(1-i_rng*dp)/sigma).^2;
    factor2 = (dt/dp)*((lambda1+lambda2)*i_rng*dp - lambda2);

    a = -0.5*factor1;
    b = 1 + factor1 - factor2;
    c = -0.5*factor1 + factor2;

    p = (0:M)' * dp;
    f_p = (mu1-mu2)*p + mu2 - 0.5*sigma^2;

    % u at state i and n=0. (or Time = T - n*dt = T)
    u0_i_n = zeros(M+1, 1);
    u1_i_n = ones(M+1, 1)*log(1-Ks);

    if x == trials
        % u_mat(:, 1) = uin;
        ttm_arr(1) = 0;
        pb_boundary_arr(1) = 1; % CHANGE
        ps_boundary_arr(1) = 1; % CHANGE
    end
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

        if x == trials
            % u_mat(:, n+1) = uin;
            ttm_arr(n+1) = n*dt;
            if sum(not(d0)) == 0
                pb_boundary_arr(n+1) = 1;
            else
                pb_boundary_arr(n+1) = p(find(not(d0), 1, 'first'));
            end

            if sum(not(d1)) == 0
                ps_boundary_arr(n+1) = 0;
            else
                ps_boundary_arr(n+1) = p(find(not(d1), 1, 'last'));
            end
        end
    end

    % if x > 1
    %     err = uin(1:2:end) - uin_prev;
    %     err_roc(x, 1) = round(sum(abs(err)), 4);
    % end
    % if x > 2
    %     err_roc(x, 2) = round(log2(err_roc(x-1,1)/err_roc(x,1)), 4);
    % end
end

time_arr = flip(ttm_arr);

% res_table = array2table(err_roc);
% res_table.Properties.VariableNames(1:2) = {'Error','Rate of Convergence'};
% res_table.Properties.RowNames = string(row_names);
% disp(res_table);

% mesh(ttm_arr, p, u_mat)
% xlabel({'Time to maturity (T-t)'});
% ylabel({'Spot Price'});
% zlabel({'American Put Option Price'});
% 
% figure()
% plot(p, u_mat(:, end))
% title('American Put Option Price vs Spot Price at time=0')
% ylabel({'Put Option Price'});
% xlabel({'Spot Price'});

figure()
plot(time_arr, pb_boundary_arr, time_arr, ps_boundary_arr)
% plot(time_arr, ps_boundary_arr)
title('Optimal Buy and Sell Boundaries')
ylabel({'p'});
xlabel({'t'});
