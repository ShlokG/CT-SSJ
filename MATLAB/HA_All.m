
% aaa
clear all; clc; close all;
dbstop if error
beep off

%% numerical inputs

% start timer
tic ;
time1 = toc ;

% baseline numerical parameters
dt   = 1.0;       % length of time period
T    = 300 / dt;  % number of time periods
Nz   = 25;        % number of productivity gridpoints
Na   = 2500;      % number of asset gridpoints
Ntot = Na * Nz;

anticipate = false; % whether to anticipate the time-0 r shock
mu         = 0.5;   % mass of agents moving out of current gridpt in transition
plot_T     = 30;    % number of periods to plot IRFs for

% Directories
fig_dir    = 'Figures/HA_';
str_append = sprintf('_Nz%d_Na%d', Nz, Na); % string to append to file name
if dt ~= 1.0
    str_append = [str_append sprintf('_dt%.1f', dt)];
end

%% define economic and numerical parameters

% preferences
gamma = 2;           % risk aversion
rho   = 0.05;        % annualized discount rate

% individual productivity process
rho_e   = 0.91; % rho for AR(1) of log(e)
sigma_e = 0.5;  % cross-sectional stdev of log income

% create idio. prod. grid & transition matrix via Rouwenhorst
[z, ~, Pi] = markov_rouwenhorst(rho_e, sigma_e, Nz);
ly         = Pi - eye(Nz);
N          = 1; % mean idiosyncratic productivity in economy

% production
d     = 0.1 ;      % annualized capital depreciation rate
alpha = 1/3 ;      % capital share in production
AgZ   = 1;         % mean aggregate productivity
rho_Z = -log(0.7); % persistence of aggregate productivity

% borrowing constraint
amin = 0;

%% grids
amax = 200;                       % maximum assets
a    = linspace(amin, amax, Na)'; % asset grid
da0  = a(2:end) - a(1:end-1);
da   = [ da0(1); da0 ];           % grid of asset steps da

% grids in asset x income space
aa   = repmat(a, 1, Nz);         % assets
daa  = repmat(da, 1, Nz);        % asset steps
zz   = repmat(z, Na, 1);         % productivity

% convergence and smoothing criteria
Delta      = 10000;        % time step smoothing for HJB
crit_S     = 1e-6;         % convergence criterion
rmin       = 1e-5;         % lower bound on possible interest rate
rmax       = rho * 0.9999; % upper bound on possible interest rate
Ir         = 300;          % maximum number of interest rate iterations
ifix       = 1;            % index where to normalize the distribution inversion
back_tol   = 1e-8;         % backward iteration convergence tolerance
back_maxit = 5000;         % backward iteration maximum iterations

% transition matrices
Ly   = kron(ly, speye(Na));
Ly_T = Ly';

time2 = toc ;
fprintf('defining parameters: %.3f s.\n', time2);

%% solve for steady-state

% normalization of right-hand-side of KFE for inversion
gRHS       = zeros(Ntot, 1);
gRHS(ifix) = 1;
gRow       = zeros(Ntot, 1);
gRow(ifix) = 1;

% initialization
dVf = zeros(Na, Nz);     % forward finite difference of value
dVb = zeros(Na, Nz);     % backward finite difference of value
c   = zeros(Na, Nz);     % consumption
r   = (rmin + rmax) / 2; % interest rate

% express wage as a function of current interest rate
w  = alpha ^ (alpha / (1 - alpha)) * (1 - alpha) * AgZ ^ (1 / (1 - alpha)) * (r + d) ^ (-alpha / (1 - alpha)) ;
ra = r * aa; % interest income
wz = w * zz; % labor income

% initial guess of value function
if gamma ~= 1
    v = ((wz + ra) .^ (1 - gamma)) / (1 - gamma) / rho;
else
    v = log(wz + ra) / rho;
end

% loop over r
for ir = 1:Ir

    % express wage as a function of current interest rate guess
    w  = alpha ^ (alpha / (1 - alpha)) * (1 - alpha) * AgZ ^ (1 / (1 - alpha)) * (r + d) ^ (-alpha / (1 - alpha)) ;
    ra = r * aa; % interest income
    wz = w * zz; % labor income

    if wz(1,1) + r * amin < 0
        disp('CAREFUL: borrowing constraint too loose');
    end

    % converge value function
    for j = 1:back_maxit
        V = v;

        % forward difference
        dVf(1:end-1, :) = (V(2:end, :) - V(1:end-1, :)) ./ da(1:end-1);
        dVf(Na, :)      = (wz(Na, :) + ra(Na, :)) .^ (-gamma);

        % backward difference
        dVb(2:Na, :) = (V(2:Na, :) - V(1:Na-1, :)) ./ da(1:Na-1);
        dVb(1, :)    = (wz(1, :) + ra(1, :)) .^ (-gamma);
       
        % consumption and savings with forward difference
        cf = dVf .^ (-1 / gamma);
        sf = wz + ra - cf;

        % consumption and savings with backward difference
        cb = dVb .^ (-1 / gamma);
        sb = wz + ra - cb;

        % consumption and derivative of value function at steady state
        c0  = wz + ra;
        dV0 = c0 .^ (-gamma);

        % indicators for upwind savings rate
        If = sf > 0;
        Ib = sb < 0;
        I0 = 1 - If - Ib;

        % consumption
        dV_Upwind = dVf .* If + dVb .* Ib + dV0 .* I0;
        c         = dV_Upwind .^ (-1 / gamma);

        % utility
        if gamma ~= 1
            u = (c .^ (1 - gamma)) / (1 - gamma);
        else
            u = log(c);
        end

        % savings finite difference matrix
        Sb = - min(sb, 0) ./ daa;
        Sm = - max(sf, 0) ./ daa + min(sb, 0) ./ daa;
        Sf =   max(sf, 0) ./ daa;
        S  = spdiags( Sm(:)               ,  0, Ntot, Ntot ) ...
           + spdiags( Sb(2:Ntot)'         , -1, Ntot, Ntot ) ...
           + spdiags([ 0 ; Sf(1:Ntot-1)' ],  1, Ntot, Ntot ) ;
        
        % matrix to invert in finite difference scheme
        M = (1 / Delta + rho) * speye(Ntot) - Ly - S;

        % invert linear system
        V = M \ (u(:) + V(:) / Delta);
        V = reshape(V, Na, Nz);

        % update
        Vchange = V - v;
        v       = V;
        dist    = max(abs(Vchange(:)));
        if dist < back_tol
            break;
        end

    end

    % update aggregates
    LT         = S' + Ly_T;               % transpose of transition matrix
    LT(ifix,:) = gRow;                    % normalize transition matrix row to make system invertible
    g          = max(real(LT \ gRHS), 0); % solve for distribution by inverting linear system
    g          = g / sum(g .* daa(:));    % normalize back to unit mass
    
    KS  = sum(g .* aa(:) .* daa(:));                       % capital supply
    KD  = (alpha * AgZ / (r + d)) ^ (1 / (1 - alpha)) * N; % capital demand
    Sav = KS - KD;                                         % net savings

    % update interest rate according to bisection
    if Sav > crit_S
        rmax = r;
        r    = 0.5 * (r + rmin);
    elseif Sav < -crit_S
        rmin = r;
        r    = 0.5 * (r + rmax);
    elseif abs(Sav) < crit_S
        break;
    end

    if ir == Ir
        fprintf('\nCould not find steady-state, r = %.4f\n', r);
        error('Could not find steady-state');
    end
end

% recompute prices for consistency
r = -d + alpha * AgZ * max(dot(aa(:), g .* daa(:)), 1e-5) ^ (alpha - 1) * N ^ (1 - alpha);
w = alpha ^ (alpha / (1 - alpha)) * (1 - alpha) * AgZ ^ (1 / (1 - alpha)) * (r + d) ^ (-alpha / (1 - alpha));

% distribution probability mass function and generator
gm = reshape(g, Na, Nz) .* daa;
L  = LT';

% get which agents are constrained
consted     = (aa == amin) .* I0 == 1;
consted_ind = find(consted(:));

% first and second derivatives of utility
up    = c .^ (-gamma);
upp   = -gamma * c .^ (-gamma - 1);
U_inv = 1 ./ upp;

time3 = toc ;
fprintf('solving steady-state: %.3f s.\n', time3-time2);



%% construct sequence-space Jacobians

% construct d/da matrix for distribution using upwinding
Sf1     = (min(sb, 0) < -1e-14) ./ daa;
Sm1     = ((max(sf, 0) > 1e-14) - (min(sb, 0) < -1e-14)) ./ daa;
Sb1     = -(max(sf, 0) > 1e-14) ./ daa;

DA_T    = spdiags(Sm1(:)             , 0 , Ntot, Ntot) ...
        + spdiags(Sb1(1:Ntot-1)'     , -1, Ntot, Ntot) ...
        + spdiags([0 ; Sf1(2:Ntot)'] , 1 , Ntot, Ntot) ;

% construct d/da matrix for values
sav_sign = sign(Sm1);  % get whether saving/dissaving for later

Sm1(1,:) = 1/da(1);
Sb1(1,:) = -1/da(1);
Sf1(1,:) = 0;

DA = -spdiags(Sm1(:), 0, Ntot, Ntot) ...
    - spdiags(Sf1(2:Ntot)', -1, Ntot, Ntot) ...
    - spdiags([0; Sb1(1:Ntot-1)'], 1, Ntot, Ntot);

% calculate other necessary values

% get asset index agent moving to via saving for later use
S_da     = -full(diag(S) / mu * dt);          % num gridpts moving via savings at each index
S_pts    = reshape(S_da, Na, Nz) .* sav_sign; % get correct sign on num gridpts moved
a_ind    = floor(S_pts) + (1:Na)';            % from change in gridpt to new gridpt moving to
sav_wt   = 1 - mod(S_pts, 1);                 % interpolation weight on left gridpt

% adjust a_ind and sav_wt so don't go out of grid
out_of_bounds_high         = a_ind >= Na;    % indices saving outside of grid
a_ind2                     = a_ind;          % store unadjusted for later use
sav_wt(out_of_bounds_high) = 0;              % move agents to last gridpt if out of bounds
a_ind(out_of_bounds_high)  = Na - 1;      

out_of_bounds_low         = a_ind <= 0;      % indices dissaving outside of grid
sav_wt(out_of_bounds_low) = 1;               % move agents to last gridpt if out of bounds
a_ind(out_of_bounds_low)  = 1;
a_ind                     = a_ind - (1:Na)'; % convert back to change in gridpts

% adjust mu at gridpts where savings would push agents out of
% bounds to ensure correct mass moving to endpoint
mu_mat = ones(Na, Nz) .* mu; % define a separate mass leaving for each gridpt

% saving too much
pts_moved                  = amax - aa(out_of_bounds_high);                        % gridpts actually moved
pts_supposed               = S_pts(out_of_bounds_high) .* daa(out_of_bounds_high); % gridpts supposed to move
mu_mat(out_of_bounds_high) = min(pts_supposed ./ pts_moved .* mu, 1);              % increase mass leaving so correct transition maintained

% repeat for case when dissaving too much
pts_moved                 = aa(out_of_bounds_low) - amin;                        % gridpts actually moved
pts_supposed              = -S_pts(out_of_bounds_low) .* daa(out_of_bounds_low); % gridpts supposed to move
mu_mat(out_of_bounds_low) = min(pts_supposed ./ pts_moved .* mu, 1);             % increase mass leaving so correct transition maintained

% convert a_ind and sav_wt into a transition matrix
% akin to LT but can move multiple asset gridpts for stability
S_stable = sparse([(1:Ntot)'; (1:Ntot)'], ...
                  [(1:Ntot)' + a_ind(:); (1:Ntot)' + a_ind(:) + 1], ...
                  [sav_wt(:); 1 - sav_wt(:)], Ntot, Ntot);

% subtract mass leaving
S_stable = (S_stable - speye(Ntot)) .* mu_mat(:) ./ dt; % multiply by mu / dt so only fraction moves

%% get values useful for solution
K = sum(gm(:) .* aa(:)); % aggregate capital
C = sum(gm(:) .* c(:));  % aggregate consumption

% aggregate shock
zeta_r    = alpha * (N / K) ^ (1 - alpha);    % zeta_r = dr/dz(0, g^{SS}). 0 if t \neq s
zeta_w    = (1 - alpha) * (N / K) ^ (-alpha); % zeta_w = dw/dz(0, g^{SS})
omega_bar = -K / N; % dw/dr

% check that income equals output
assert(abs(w * N + (r + d) * K - AgZ * N ^ (1 - alpha) * K ^ alpha) < 1e-3)

time4 = toc ;
fprintf('prepping for Jacobians: %.3f s.\n', time4-time3);

%% step 1: calculate change in value from future shock and consumption at time 0
% solve for phi_t by iterating forward
phi_r          = zeros(Na, Nz, T); % initializations
phi_r(:, :, 1) = aa .* up;         % get \vp_0: income change from r shock * u'(c)
phi_w          = zeros(Na, Nz, T);
phi_w(:, :, 1) = zz .* up;         % income change from wage shock * u'(c)

% apply constraints
phi_r_assign          = phi_r(:, :, 1);         % array to be changed
phi_w_assign          = phi_w(:, :, 1);

phi_r_2on             = phi_r_assign(2:end, :); % for each constrained agent, need agent at asset pt above
phi_w_2on             = phi_w_assign(2:end, :);
consted_slice         = consted(1:end-1, :);    % which agents constrained

upp_inc_r             = upp .* aa;
upp_inc_w             = upp .* zz;

phi_r_assign(consted) = phi_r_2on(consted_slice) - da(1) * upp_inc_r(consted); % assign constrained agents to value of agents at asset pt above
phi_w_assign(consted) = phi_w_2on(consted_slice) - da(1) * upp_inc_w(consted);

phi_r(:, :, 1)        = phi_r_assign;           % assign in actual array
phi_w(:, :, 1)        = phi_w_assign;

% iteration
for t = 1:T-1
    % save last period's phi
    phi_rt = phi_r(:, :, t);
    phi_wt = phi_w(:, :, t);

    % transition to next period
    phi_r(:, :, t + 1) = phi_rt + dt * (phi_rt * ly' + reshape(S_stable * phi_rt(:), Na, Nz));
    phi_w(:, :, t + 1) = phi_wt + dt * (phi_wt * ly' + reshape(S_stable * phi_wt(:), Na, Nz));
    
    % adjust constrained so phi for constrained equals that for assets just
    % above constrained
    phi_r_assign          = phi_r(:, :, t + 1);       % array to be changed
    phi_w_assign          = phi_w(:, :, t + 1);

    phi_r_2on             = phi_r_assign(2:end, :);   % for each constrained agent, need agent at asset pt above
    phi_w_2on             = phi_w_assign(2:end, :);
    consted_slice         = consted(1:end-1, :);      % which agents constrained
    
    phi_r_assign(consted) = phi_r_2on(consted_slice); % assign constrained agents to value of agents at asset pt above
    phi_w_assign(consted) = phi_w_2on(consted_slice);

    phi_r(:, :, t + 1)    = phi_r_assign;             % assign in actual array
    phi_w(:, :, t + 1)    = phi_w_assign;
end

% reshape into vector at each time for later matrix multiplication
phi_r = reshape(phi_r, Ntot, T);
phi_w = reshape(phi_w, Ntot, T);

% derivative with respect to assets
dphi_da_r = DA * phi_r;
dphi_da_w = DA * phi_w;

% adjust time-0 for ex-post shock
if ~anticipate
    dphi_da_r(:, 1) = (DA * up(:)) .* aa(:);
    dphi_da_w(:, 1) = (DA * up(:)) .* zz(:);
end

% calculate change in consumption at time 0
rho_T     = exp(-rho * (0:T-1) * dt); % discounting of future
c_r       = dphi_da_r .* (U_inv(:) .* gm(:)) .* rho_T; % consumption response to a future r shock
c_w       = dphi_da_w .* (U_inv(:) .* gm(:)) .* rho_T; % consumption response to a future w shock

% adjust distribution for consumption adjustment at time 0
if ~anticipate
    c_prime = DA * c(:);             % c'(a)
    aa_g    = aa .* gm;              % a*g(x) for each type x=(a,z)
    c_r(:, 1)  = aa_g(:) .* c_prime; % adjusted time-0 consumption
end

% total change in distribution at each time
D_r       = DA_T * c_r;
D_r(:, 1) = D_r(:, 1) - DA_T * (aa(:) .* gm(:)); % adjust for change in distribution at time 0 directly from income change

D_w       = DA_T * c_w;
D_w(:, 1) = D_w(:, 1) - DA_T * (zz(:) .* gm(:)); % adjust for change in distribution at time 0 directly from income change

% aggregate change in consumption
C_r = sum(c_r);
C_w = sum(c_w);

time5 = toc ;
fprintf('Step 1 in Jacobians: policy functions: %.3f s.\n', time5-time4);

%% calculate expectation vectors E (dy_t/dg(x)_0)
% change in output from change in mass at x at time 0
% unders steady state policy function

% initialize
E_r = zeros(T, Na, Nz);
E_w = zeros(T, Na, Nz);
E_K = zeros(T, Na, Nz);
E_C = zeros(T, Na, Nz);

% effect at time 0
E_r(1, :, :) = AgZ * alpha * (1 - alpha) * (N / K) ^ (1 - alpha) * (zz ./ N - aa ./ K);
E_w(1, :, :) = E_r(1, :, :) * omega_bar;
E_K(1, :, :) = aa;
E_C(1, :, :) = c;

% iteration
% equivalent to E_t = T_t E_0 in the paper
for t = 1:(T-1)
    % save last period's E
    E_rt = squeeze(E_r(t, :, :));
    E_wt = squeeze(E_w(t, :, :));
    E_Kt = squeeze(E_K(t, :, :));
    E_Ct = squeeze(E_C(t, :, :));

    % exogenous idiosyncratic productivity transition and endogenous savings transition
    E_r(t + 1, :, :) = E_rt + dt * ( E_rt * ly' + reshape(S_stable * E_rt(:), Na, Nz) ) ;
    E_w(t + 1, :, :) = E_wt + dt * ( E_wt * ly' + reshape(S_stable * E_wt(:), Na, Nz) ) ;
    E_K(t + 1, :, :) = E_Kt + dt * ( E_Kt * ly' + reshape(S_stable * E_Kt(:), Na, Nz) ) ;
    E_C(t + 1, :, :) = E_Ct + dt * ( E_Ct * ly' + reshape(S_stable * E_Ct(:), Na, Nz) ) ;
end

E_r = reshape(E_r, T, Ntot);
E_w = reshape(E_w, T, Ntot);
E_K = reshape(E_K, T, Ntot);
E_C = reshape(E_C, T, Ntot);

time6 = toc ;
fprintf('Step 2 in Jacobians: expectation functions: %.3f s.\n', time6-time5);


%% jacobian matrices
% compute fake news kernel, F_{t,s} = E_t^* D_s
F_rr = E_r * D_r; % r shock on r
F_rK = E_K * D_r; % r shock on K
F_rC = E_C * D_r; % r shock on C

F_ww = E_w * D_w; % w shock on w
F_wK = E_K * D_w; % w shock on K
F_wC = E_C * D_w; % w shock on C

% compute the Jacobian
% initialization
J_rr = zeros(T, T); % jacobian of r to a r shock
J_rK = zeros(T, T); % jacobian of K to a r shock
J_rC = zeros(T, T); % jacobian of C to a r shock

J_ww = zeros(T, T); % jacobian of w to a w shock
J_wK = zeros(T, T); % jacobian of K to a w shock
J_wC = zeros(T, T); % jacobian of C to a w shock

% get the first column from change in dist.
J_rr(2:end, 1) = F_rr(1:end-1, 1);
J_rK(2:end, 1) = F_rK(1:end-1, 1);
J_rC(2:end, 1) = F_rC(1:end-1, 1);

J_ww(2:end, 1) = F_ww(1:end-1, 1);
J_wK(2:end, 1) = F_wK(1:end-1, 1);
J_wC(2:end, 1) = F_wC(1:end-1, 1);

% consumption is a control so need to account for change in behavior
% get change in policy function consumption
J_rC(1, :) = C_r;
J_wC(1, :) = C_w;

% jacobian by adding fake news element to diagonal element 
for t = 2:T
    J_rr(2:end, t) = J_rr(1:end-1, t - 1) + dt * F_rr(2:end, t);
    J_rK(2:end, t) = J_rK(1:end-1, t - 1) + dt * F_rK(2:end, t);
    J_rC(2:end, t) = J_rC(1:end-1, t - 1) + dt * F_rC(2:end, t);

    J_ww(2:end, t) = J_ww(1:end-1, t - 1) + dt * F_ww(2:end, t);
    J_wK(2:end, t) = J_wK(1:end-1, t - 1) + dt * F_wK(2:end, t);
    J_wC(2:end, t) = J_wC(1:end-1, t - 1) + dt * F_wC(2:end, t);
end

% adjust by dt to convert into operator
J_rr = J_rr * dt ; 
J_rK = J_rK * dt ; 
J_rC = J_rC * dt ; 

J_ww = J_ww * dt ; 
J_wK = J_wK * dt ; 
J_wC = J_wC * dt ; 


time7 = toc ;
fprintf('Step 3 & 4 in Jacobians: fake news and Jacobians: %.3f s.\n', time7-time6);

%% impulse responses to aggregate shocks

% 1% TFP shock
timeZ = linspace(0, (T - 1) * dt, T);
Z_impulse = 0.01 .* exp(-rho_Z * timeZ);

% write system of eqs in matrix form, w/ r and w as 1 vector:
% p-hat = [r-hat;w-hat] = [J_rr, J_rw; J_wr, J_ww] *
% p-hat + [zeta_r I, 0; 0, zeta_w I]z-hat
full_jac = [J_rr, J_ww ./ omega_bar; omega_bar * J_rr, J_ww];

% repeat z_hat for each price and multiply by dp/dz
% TO-DO: MAKE THIS MORE CONCISE
z_hat              = repmat(Z_impulse', 2, 1) ;
z_hat(1:T)         = z_hat(1:T) * zeta_r; % TFP shock multiplied by direct effect on r
z_hat((T + 1):end) = z_hat((T + 1):end) * zeta_w; % TFP shock multiplied by direct effect on w

% solve system of equations to get r and w
p_hat = (eye(2 * T) - full_jac) \ z_hat;
dr = p_hat(1:T);
dw = p_hat((T + 1):end);

% get implied capital and consumption from Jacobians
dK = (J_rK * dr + J_wK * dw);
dC = (J_rC * dr + J_wC * dw);

time8 = toc ;
fprintf('Step 5 in Jacobians: impulse responses: %.3f s.\n', time8-time7);


%% plot jacobian columns
cols = [1, round(100/dt)+1, round(200/dt)+1];
Jacobian_Plots(J_rr, cols, 'r', 'r', dt, fig_dir, str_append)
Jacobian_Plots(J_rK / K, cols, 'K', 'r', dt, fig_dir, str_append)
Jacobian_Plots(J_rC / C, cols, 'C', 'r', dt, fig_dir, str_append)

Jacobian_Plots(J_ww, cols, 'w', 'w', dt, fig_dir, str_append)
Jacobian_Plots(J_wK / K, cols, 'K', 'w', dt, fig_dir, str_append)
Jacobian_Plots(J_wC / C, cols, 'C', 'w', dt, fig_dir, str_append)

%% plot impulse responses
plot_t = min(plot_T / dt, T);
IRFs_fn(dr(1:plot_t) .* 100, 'r', 'Z', dt, fig_dir, str_append)
IRFs_fn(dw(1:plot_t) .* 100, 'w', 'Z', dt, fig_dir, str_append)
IRFs_fn(dK(1:plot_t) ./ K .* 100, 'K', 'Z', dt, fig_dir, str_append)
IRFs_fn(dC(1:plot_t) ./ C .* 100, 'C', 'Z', dt, fig_dir, str_append)

time9 = toc ;
fprintf('Printing and saving figures: %.3f s.\n', time9-time8);

