% Finite Difference for optimal stopping American Put Option

T = 1;
r = 0.01;
sigma = 0.2;
K = 100;

R = 400;
N = 4; M = 4;

trials =7;
err_roc = NaN(trials,2);
row_names = cell(1, trials);

u_mat = NaN(M*2^trials+1, N*2^trials+1);
ttm_arr = NaN(N*2^trials+1, 1);
px_boundary_arr = NaN(N*2^trials+1, 1);

for x = 1:trials
    if x > 1
        uin_prev = uin;
        px_prev = px;
    end
    N = N*2;
    M = M*2;
    row_names(x) = compose('%dx%d',N,M);

    dt = T/N;
    dy = R/M;
    % N*dt = T; M*dy = R
    i_rng = (1:M-1)';
    a = -0.5*dt*(sigma^2)*i_rng.^2;
    b = 1 + dt*(r + r*i_rng + (sigma*i_rng).^2);
    c = dt*(-r*i_rng - 0.5*(sigma*i_rng).^2);
    temp = reshape([a; b; c], M - 1, 3);
    Ah = spdiags(temp, [-1, 0, 1], M - 1, M - 1);
    % u at state i and time n=0
    px = (0:M)' * dy;
    intrinsic_value = max(K-px, 0);
    uin = intrinsic_value;
    if x == trials
        u_mat(:, 1) = uin;
        ttm_arr(1) = 0;
        px_boundary_arr(1) = K;
    end
    for n = 1:N
        Bhn = uin(2:end-1);
        Bhn(1) = Bhn(1) - a(1)*K*exp(-r*(T-n*dt));
        % uin_ex = Ah\Bhn;
        uin_ = TDMAsolver(a,b,c,Bhn)';
        d = (uin_ - intrinsic_value(2:end-1))<0;
        u1 = d.*intrinsic_value(2:end-1);
        u2 = not(d).*uin_;
        uin_temp = u1 + u2;
        uin = [K; uin_temp; 0];

        if x == trials
            u_mat(:, n+1) = uin;
            ttm_arr(n+1) = n*dt;
            px_boundary_arr(n+1) = px(find(d, 1, 'last'));
        end
    end

    if x > 1
        err = uin(1:2:end) - uin_prev;
        err_roc(x, 1) = round(sum(abs(err)), 4);
    end
    if x > 2
        err_roc(x, 2) = round(log2(err_roc(x-1,1)/err_roc(x,1)), 4);
    end
end

time_arr = flip(ttm_arr);

res_table = array2table(err_roc);
res_table.Properties.VariableNames(1:2) = {'Error','Rate of Convergence'};
res_table.Properties.RowNames = string(row_names);
disp(res_table);

mesh(ttm_arr, px, u_mat)
xlabel({'Time to maturity (T-t)'});
ylabel({'Spot Price'});
zlabel({'American Put Option Price'});

figure()
plot(px, u_mat(:, end))
title('American Put Option Price vs Spot Price at time=0')
ylabel({'Put Option Price'});
xlabel({'Spot Price'});

figure()
plot(time_arr, px_boundary_arr)
title('Free Boundary condition vs Time Index')
ylabel({'Spot Price'});
xlabel({'Time'});
