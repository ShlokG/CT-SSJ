function [y, pi, Pi] = markov_rouwenhorst(rho, sigma, N)
    % Rouwenhorst method analog to markov_tauchen

    if nargin < 3
        N = 7;
    end

    % parametrize Rouwenhorst for n=2
    p = (1 + rho) / 2;
    Pi = [p, 1 - p; 1 - p, p];

    % implement recursion to build from n=3 to n=N
    for n = 3:N
        P1 = zeros(n, n);
        P2 = zeros(n, n);
        P3 = zeros(n, n);
        P4 = zeros(n, n);
        P1(1:end-1, 1:end-1) = p * Pi;
        P2(1:end-1, 2:end) = (1 - p) * Pi;
        P3(2:end, 1:end-1) = (1 - p) * Pi;
        P4(2:end, 2:end) = p * Pi;
        Pi = P1 + P2 + P3 + P4;
        Pi(2:end-1, :) = Pi(2:end-1, :) / 2;
    end

    % invariant distribution and scaling
    pi = stationary(Pi);
    s = linspace(-1, 1, N);
    s = s * (sigma / sqrt(variance(s, pi)));
    y = exp(s) / sum(pi .* exp(s));

    return
end

function pi = stationary(Pi)
    % Calculate the stationary distribution of a transition matrix Pi
    [V, D] = eig(Pi');
    [~, idx] = min(abs(diag(D) - 1));
    pi = V(:, idx);
    pi = pi / sum(pi);
    pi = pi';
end

function var = variance(s, pi)
    % Calculate the variance of a distribution
    mean_s = sum(pi .* s);
    var = sum(pi .* (s - mean_s).^2);
end