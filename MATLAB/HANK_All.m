
% aaa
clear all; clc; close all;
dbstop if error
beep off

%% numerical inputs

% start timer
tic ;
time1 = toc ;

% baseline numerical parameters
dt   = 1.0;      % length of time period
T    = 300 / dt; % number of time periods
Nz   = 25;       % number of productivity gridpoints
Na   = 2500;     % number of asset gridpoints
Ntot = Na * Nz;

anticipate = false; % whether to anticipate the time-0 r shock
mu         = 0.5;   % mass of agents moving out of current gridpt in transition
plot_T     = 30;    % number of periods to plot IRFs for

% Directories
fig_dir = 'Figures';
str_append = sprintf('_Nz%d_Na%d', Nz, Na);
if dt ~= 1.0
    str_append = [str_append sprintf('_dt%.1f', dt)];
end

%% define economic and numerical parameters
% preferences
gamma = 2;          % risk aversion 
rho   = -log(0.94); % annualized discount rate

% income
theta = 0.181; % progressivity of HSV
Z     = 0.471; % Y - T

% individual productivity process
rho_e   = 0.91;               % rho for AR(1) of log(e)
sigma_e = (1 - theta) * 0.92; % cross-sectional stdev of income

% create idio. prod. grid & transition matrix via Rouwenhorst
[z, ~, Pi] = markov_rouwenhorst(rho_e, sigma_e, Nz);
ly         = Pi - eye(Nz);

% output
AgZ = 1.0;       % mean aggregate productivity
Nss = 1.0;       % steady-state labor
Yss = AgZ * Nss; % steady-state output

% labor market
mu_w    = 1.1;                                        % wage markup
kappa_w = 0.03;                                       % wage flexibility
xi      = 1;                                          % labor supply elasticity
psi     = mu_w / (mu_w - 1) / kappa_w * Nss^(1 + xi); % wage adjustment costs

% Interest Rates, Tax
phi = 1.0;                     % Taylor Rule coefficient on inflation

% Steady State Calibration
G_share = 0.2;                 % G/Y
T_share = (Yss - Z) / Yss;     % T/Y
w       = (1 - T_share) * Yss; % post-tax income

% borrowing constraint
amin = 0;

%% grids
amax = 1000; % maximum assets
% if too few asset gridpts, lower amax to shrink asset step size
if Na <= 100 && Nz <= 2
    amax = 400;
end

a    = linspace( amin , amax , Na )'; % linear Asset Grid
da0  = a(2:end) - a(1:end-1);
da   = [ da0(1); da0 ];               % grid of asset steps da

% grids in asset x income space
aa   = repmat(a, 1, Nz);  % assets
daa  = repmat(da, 1, Nz); % asset steps
zz   = repmat(z, Na, 1);  % productivity

wz = w * zz; % labor income

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

% shock processes
rho_G = -log(0.8); % persistence of G shock
rho_r = 0.8;       % persistence of r shock
rho_b = -log(0.5); % beta in paper (how quickly debt paid off) 

time2 = toc ;
fprintf('defining parameters: %.3f s.\n', time2);

%% solve for steady-state

% normalization of right-hand-side of KFE for inversion
gRHS       = zeros(Ntot, 1);
gRHS(ifix) = 1;
gRow       = zeros(Ntot, 1);
gRow(ifix) = 1;

% initialization
r   = (rmin + rmax) / 2; % interest rate

% loop over r
for ir = 1:Ir
    ra = r * aa; % interest income

    if wz(1,1) + r * amin < 0
        disp('CAREFUL: borrowing constraint too loose');
    end

    % initializations
    dVf = zeros(Na, Nz);     % forward finite difference of value
    dVb = zeros(Na, Nz);     % backward finite difference of value
    c   = zeros(Na, Nz);     % consumption

    % initial guess
    if gamma ~= 1
        v = ((wz + ra) .^ (1 - gamma)) / (1 - gamma) / rho;
    else
        v = log(wz + ra) / rho;
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
    
    KS  = sum(g .* aa(:) .* daa(:));      % capital supply
    KD = (T_share - G_share) * Yss / r;   % capital demand
    Sav = KS - KD;                        % net savings

    % update interest rate according to bisection
    if Sav > crit_S
        rmax = r;
        r = 0.5 * (r + rmin);
    elseif Sav < -crit_S
        rmin = r;
        r = 0.5 * (r + rmax);
    elseif abs(Sav) < crit_S
        break;
    end

    if ir == Ir
        fprintf('\nCould not find steady-state, r = %.4f\n', r);
        error('Could not find steady-state');
    end
end

% recompute prices for consistency
B_share = sum(g .* daa(:) .* aa(:)); % aggregate assets
r       = (T_share - G_share) * Yss / B_share;

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
a_ind    = floor(S_pts) + (1:Na)';            % from change in gridpts to new gridpt moving to
sav_wt   = 1 - mod(S_pts, 1);                 % interpolation weight on left gridpt

% adjust a_ind and sav_wt so don't go out of grid
out_of_bounds_high         = a_ind >= Na;    % indices saving outside of grid
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
pts_moved                  = amax - aa(out_of_bounds_high);                        % assets actually moved
pts_supposed               = S_pts(out_of_bounds_high) .* daa(out_of_bounds_high); % assets supposed to move
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

% subtract the identity matrix
S_stable = (S_stable - speye(Ntot)) .* mu_mat(:) ./ dt; % multiply by mu / dt so only fraction moves

%% get values useful for solution
Capital = sum(gm(:) .* aa(:));  % aggregate capital
C       = sum(gm(:) .* c(:));   % aggregate consumption
Lab     = sum(gm .* zz, "all"); % aggregate labor

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
    phi_r(:, :, t + 1) = phi_rt + dt * ( phi_rt * ly' + reshape(S_stable * phi_rt(:), Na, Nz) ) ;
    phi_w(:, :, t + 1) = phi_wt + dt * ( phi_wt * ly' + reshape(S_stable * phi_wt(:), Na, Nz) ) ;
    
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
E_K = zeros(T, Na, Nz);
E_C = zeros(T, Na, Nz);

% effect at time 0
E_K(1, :, :) = aa;
E_C(1, :, :) = c;

% iteration
% equivalent to E_t = T_t E_0 in the paper
for t = 1:(T-1)
    % save last period's E
    E_Kt = squeeze(E_K(t, :, :));
    E_Ct = squeeze(E_C(t, :, :));

    % exogenous idiosyncratic productivity transition
    E_K(t + 1, :, :) = E_Kt + dt * ( E_Kt * ly' + reshape(S_stable * E_Kt(:), Na, Nz) ) ;
    E_C(t + 1, :, :) = E_Ct + dt * ( E_Ct * ly' + reshape(S_stable * E_Ct(:), Na, Nz) ) ;
end

E_K = reshape(E_K, T, Ntot);
E_C = reshape(E_C, T, Ntot);

time6 = toc ;
fprintf('Step 2 in Jacobians: expectation functions: %.3f s.\n', time6-time5);

%% jacobian matrices
% compute fake news kernel, F_{t,s} = E_t^* D_s
F_rK = E_K * D_r; % r shock on K
F_rC = E_C * D_r; % r shock on C

F_wK = E_K * D_w; % w shock on K
F_wC = E_C * D_w; % w shock on C

% compute the Jacobian
% initialization
J_rK = zeros(T, T); % jacobian of K to a r shock
J_rC = zeros(T, T); % jacobian of C to a r shock

J_wK = zeros(T, T); % jacobian of K to a w shock
J_wC = zeros(T, T); % jacobian of C to a w shock

% get the first column from change in dist.
J_rK(2:end, 1) = F_rK(1:end-1, 1);
J_rC(2:end, 1) = F_rC(1:end-1, 1);

J_wK(2:end, 1) = F_wK(1:end-1, 1);
J_wC(2:end, 1) = F_wC(1:end-1, 1);

% consumption is a control so need to account for change in behavior
% get change in policy function consumption
J_rC(1, :) = C_r;
J_wC(1, :) = C_w;

% jacobian by adding fake news element to diagonal element 
for t = 2:T
    J_rK(2:end, t) = J_rK(1:end-1, t - 1) + dt * F_rK(2:end, t);
    J_rC(2:end, t) = J_rC(1:end-1, t - 1) + dt * F_rC(2:end, t);
    J_wK(2:end, t) = J_wK(1:end-1, t - 1) + dt * F_wK(2:end, t);
    J_wC(2:end, t) = J_wC(1:end-1, t - 1) + dt * F_wC(2:end, t);
end

% adjust by dt to convert into operator
J_rK = J_rK * dt; 
J_rC = J_rC * dt; 

J_wK = J_wK * dt; 
J_wC = J_wC * dt; 

time7 = toc ;
fprintf('Step 3 & 4 in Jacobians: fake news and Jacobians: %.3f s.\n', time7-time6);


%% impulse responses to aggregate shocks

% shocks
T2 = T - 10;                         % reduce T since matrix non-invertible otherwise
dG = exp(-rho_G .* (0:T2-1)' .* dt); % shock to gov't spending
dr = (rho_r .^ (0:T2-1)') .^ dt;     % shock to interest rates

% calculate value for bonds
dB = zeros(T2, 1);
for t = 1:T2
    if t > 1
        dB_lag = dB(t-1);
    else
        dB_lag = 0;
    end
    dB(t) = (1 - rho_b * dt) * (dB_lag + dG(t));
end

% calculate corresponding tax process to balance budget
dB_lag = [0; dB(1:end-1)];
dT = dG + (1 + r) * dB_lag - dB;

% K matrix needed to make jacobian invertible
q = (1 + r) .^ -((0:T-1) .* dt);
K = triu(toeplitz(-q), 1);

% calculate general equilibrium jacobians and impulse responses
if phi == 1.0 % passive monetary policy
    % get invertible jacobian of Y w.r.t. G
    A  = K * (eye(T) - J_wC);            % asset jacobian
    cM = A(1:T2, 1:T2) \ K(1:T2, 1:T2);  % jacobian of Y w.r.t. G

    % set Y Jacobians
    J_YG   = cM;
    J_YT   = -cM * J_wC(1:T2, 1:T2);
    J_Yeps = cM * J_rC(1:T2, 1:T2);
    J_reps = eye(T);

    dY_dG_ge = cM * (dG - J_wC(1:T2, 1:T2) * dT);                        % Y response to G shock, accounting for induced change in T
    dT_bal   = dr * Capital;                                             % when r changes, T must too to balance budget
    dY_dr_ge = cM * (J_rC(1:T2, 1:T2) * dr - J_wC(1:T2, 1:T2) * dT_bal); % Y response to r shock, accounting for induced change in R

% expressions more complicated if active monetary policy
else
    rho_disc = triu(toeplitz(exp(-rho) .^ ((0:T-1) .* dt)), 0); % discounting of future
    % forward matrix since inflation next period matters when discretizing
    Fmat = eye(T);
    Fmat = [Fmat(2:end, :); zeros(1, T)];
    phiIF = (phi * eye(T) - Fmat);

    % inflation response to each variable
    J_piY = rho_disc * kappa_w * ((1 + xi) / Yss + gamma / C - 1 / (Yss * (1 - T_share))); % contemporaneous partial equilibrium response of inflation to change in Y
    J_piZ = -rho_disc * kappa_w * (1 + xi) / AgZ;                                          % contemporaneous partial equilibrium response of inflation to change in productivity
    J_piT = rho_disc * kappa_w * 1 / (Yss * (1 - T_share));                                % contemporaneous partial equilibrium response of inflation to change in T
    J_piG = -rho_disc * kappa_w * gamma / C;                                               % contemporaneous partial equilibrium response of inflation to change in G
    J_pir = zeros(T);                                                                      % contemporaneous partial equilibrium response of inflation to change in Y

    J_Tr  = eye(T) * Capital; % taxes increase to keep debt unchanged

    % GE jacobian of Y and r to a G shock
    B = eye(T) - phiIF * (J_pir + J_piT * J_Tr + J_piY * ((eye(T) - J_wC) \ (J_rC - J_wC * J_Tr)));
    J_rG = B \ (phiIF * (J_piY / (eye(T) - J_wC) + J_piG));
    J_YG = (eye(T) - J_wC) \ (J_rC * J_rG - J_wC * J_Tr * J_rG + eye(T));

    % GE jacobian of Y and r to a T shock
    J_rT = B \ (phiIF * (J_piT - J_piY / (eye(T) - J_wC) * J_wC));
    J_YT = (eye(T) - J_wC) \ ((J_rC - J_wC * J_Tr) * J_rT - J_wC);

    % GE jacobian of Y and r to aggregate productivity shock
    J_rZ = B \ (phiIF * J_piZ);
    J_YZ = (eye(T) - J_wC) \ (J_rC * J_rZ - J_wC * J_Tr * J_rZ);
    
    % GE jacobian of Y and r to interest rate shock
    J_reps = B \ eye(T);
    J_Yeps = (eye(T) - J_wC) \ ((J_rC - J_wC * J_Tr) * J_reps);

    % impulse responses for 1) G and corresponding T shocks and 2) r shock
    dY_dG_ge = J_YG(1:T2, 1:T2) * dG + J_YT(1:T2, 1:T2) * dT;
    dY_dr_ge = J_Yeps(1:T2, 1:T2) * dr;
    
    dr_dG_ge = J_rG(1:T2, 1:T2) * dG + J_rT(1:T2, 1:T2) * dT;
    dr_dr_ge = J_reps(1:T2, 1:T2) * dr;
end

%% plot partial equilibrium MPC matrix
columns_to_plot = [1, round(100/dt)+1, round(200/dt)+1];
default_colors = get(gca, 'ColorOrder');

for i = 1:length(columns_to_plot)
    col = columns_to_plot(i);
    plot(0:dt:(T-1)*dt, J_wC(:, col) / C, 'DisplayName', strcat('s=', string(round((col - 1) * dt))), ...
        'Color', default_colors(i, :), 'LineWidth', 3);
    if i == 1
        legend('show', 'Location', 'best', 'FontSize', 16);
    end
    hold on
end

xlabel('Year', 'FontSize', 14);
ylabel('p.p. deviation from SS', 'FontSize', 14);
set(gca, 'FontSize', 12);

saveas(gcf, fullfile(fig_dir, ['HANK_MPC' str_append '.pdf']));
hold off;

%% plot impulse responses of Y to G and r shocks
set(groot, 'defaultLegendFontSize', 16);
plot_T = min(T2 * dt, plot_T);

plot(0:dt:(plot_T - dt), dY_dG_ge(1:floor(plot_T / dt)), 'DisplayName', 'Continuous (surprise)', ...
    'LineWidth', 3, 'Color', default_colors(1, :));
legend('show', 'Location', 'best', 'FontSize', 16);
xlabel('Year', 'FontSize', 14);
ylabel('p.p. deviation from SS', 'FontSize', 14);
set(gca, 'FontSize', 12);
saveas(gcf, fullfile(fig_dir, ['HANK_IRF_YG' str_append '.pdf']));
hold off;

plot(0:dt:(plot_T - dt), dY_dr_ge(1:floor(plot_T / dt)), 'DisplayName', 'Continuous (surprise)', ...
    'LineWidth', 3, 'Color', default_colors(1, :));
legend('show', 'Location', 'best', 'FontSize', 16);
xlabel('Year', 'FontSize', 14);
ylabel('p.p. deviation from SS', 'FontSize', 14);
set(gca, 'FontSize', 12);
saveas(gcf, fullfile(fig_dir, ['HANK_IRF_Yr' str_append '.pdf']));
hold off;

% Note: If IRFs have the opposite sign to expected in the first period, increase T




