% Script Model-Based Inverse Reinforcement Learning
% This script obtains the objective function of optimal LQR policies
% Drone dynamics assumed to be linear

clc
close all
clear all

global A
global B
global m
global g
global Kreal

% Seed to obtain the same results
rng(800)

% Parameters of the drone
g = 9.81;
m = 0.467;

% Initial hover condition
x0 = [0.5, 0.3, m*g, 0, 0, 0];

% Drone Linear Dynamics
A = [0*eye(3) eye(3);
    0*eye(3) 0*eye(3)]; % Matrix A

B = [zeros(3,3);
    0 -g 0;
    g 0 0;
    0 0 1/m]; % Matrix B

% Real Weight Matrices
Q0 = randn(6,6);
Q0 = 1/2*(Q0.'*Q0);
Qreal = Q0;%diag([1, 1, 1, 5, 5, 5]);
Rreal = 10*eye(3);

% Solve ARE to obtain the real P and gain K
[Preal, ~, Kreal] = care(A, B, Qreal, Rreal);

% Evaluate the controller and collect data
[t, x] = ode45(@(t,x)ode_file(t, x, xd(t)), 0:0.02:180, x0);

% Corrupt data with Gaussian noise
% Large level of noise affects the algorithm
X = x.' + 0e-4*std(x.', [], 'all').*randn(size(x.'));
Xd = [];
for i = 1:length(x)
    % Tracking error
    e(:,i) = xd(i*0.02) - x(i, :).';

    % Linear Control policy
    Ups(:,i) = Kreal*e(:,i);

    % Real Objective function measurement
    Xi(i) = e(:,i).'*Qreal*e(:,i) + Ups(:,i).'*Rreal*Ups(:,i);

    % Store the desired reference into a matrix
    Xd = [Xd xd(i*0.02)];
end

% Similar to DMD: Apply SVD to approximate the real gain K
[U, Sig, V] = svd(Xd - X, 'econ');

% Check that the singular values are greater than a threshold
thresh = 1e-10;
rtil = length(find(diag(Sig) > thresh));

% Consider only the dimensions that satisfy the threshold
Util = U(:, 1:rtil);
Sigtil = Sig(1:rtil, 1:rtil);
Vtil = V(:, 1: rtil);

% Estimated gain
Kest = Ups*Vtil*pinv(Sigtil)*Util.';

% Initialize random weight matrices for the Model-Based IRL
Q(:, :, 1) = 10*eye(6);
R(:, :, 1) = 1*eye(3);

% Find an initial stabilizable control policy
[~, ~, K(:, :, 1)] = care(A, B, Q(:, :, 1), R(:, :, 1));

% Parametrize the objective function in terms of the quadratic error
XX = [];
for i = 1:length(X)
    XX = [XX; kron(Xd(:,i) - x(i,:).', Xd(:,i) - x(i,:).').'];
end

n = length(A);
episodes = 5000;

% Main Loop of the model-based IRL
for i = 1: episodes
    % Solve a Least-squared Lyapunov Recursion model
    W = -(K(:, :, i) - Kest).'*R(:, :, i)*(K(:, :, i) - Kest) + ...
        Q(:, :, i) + K(:, :, i).'*R(:, :, i)*K(:, :, i);
    vecW = reshape(-W, [], 1);
    vecP = pinv(kron(eye(n), (A - B*K(:, :, i))).' + ...
        kron((A - B*K(:, :, i)), eye(n)).')*vecW;

    % New kernel matrix P(i)
    P(:, :, i) = reshape(vecP, n, n);

    % New improved control gain K(i+1)
    K(:, :, i + 1) = R(:, :, i)\B.'*P(:, :, i);     
    
    % Build matrices for the IRL
    H = A.'*P(:, :, i) + P(:, :, i)*A;
    vecH = reshape(-H, [] , 1);

    % Concatenate the Riccati equation and objective function data
    Omega = [vecH; pinv(XX)*Xi.'];
  
    Psi = [eye(n^2) -kron(K(:, :, i + 1), K(:, :, i + 1)).';
        eye(n^2) kron(Kest.', Kest.')];  
    
    % Solve a Least squares algorithm
    Theta(:, i) = pinv(Psi)*Omega;

    % New improved weight matrices Q(i+1) and R(i+1)
    Q(:, :, i + 1) = reshape(Theta(1:n^2, i), size(Q(:, :, i)));
    R(:, :, i + 1) = reshape(Theta(n^2 + 1: end, i), size(R(:,:,i)));


    % Check if the norm between actual and previous gain is less than
    % a threshold: if true stop the loop, otherwise continue

    if norm(K(:,:,i + 1) - K(:, :, i)) <= 1e-8
        break
    end
end

Kreal
Kapprox = K(:,:,end)

Qreal
Qapprox = Q(:,:,end)

Rreal
Rapprox = R(:,:, end)

for i = 1 : size(P,3) - 1
    episode(i) = i-1;
    Pnorm(i) = norm(P(:, :, i+1) - P(:, :, i));
    Qnorm(i) = norm(Q(:, :, i+1) - Q(:, :, i));
    Rnorm(i) = norm(R(:, :, i+1) - R(:, :, i));
    Knorm(i) = norm(K(:, :, i+1) - K(:, :, i));
end

figure(1)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
plot(episode(1:21), Knorm(1:21), 'linewidth', 2);
hold on
plot(episode(1:21), Pnorm(1:21), 'linewidth', 2);
plot(episode(1:21), Qnorm(1:21), 'linewidth', 2);
plot(episode(1:21), Rnorm(1:21), 'linewidth', 2);
set(gca, 'fontsize', 16)
grid on
xlabel({'Episodes $i$'},'interpreter','latex');
ylabel({'$\|\cdot\|$'},'interpreter','latex');
legend({'$\|\textbf{K}_{i+1}-\textbf{K}_i\|$', ...
    '$\|\textbf{P}_{i+1}-\textbf{P}_i\|$', ...
    '$\|\textbf{Q}_{i+1}-\textbf{Q}_i\|$', ...
    '$\|\textbf{R}_{i+1}-\textbf{R}_i\|$'},'interpreter','latex');

figure(2)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
plot(episode(1:101), Theta(:, 1:101), 'linewidth', 2);
set(gca, 'fontsize', 16)
grid on
xlabel({'Episodes $i$'},'interpreter','latex');
ylabel({'$\Theta_i$ values'},'interpreter','latex');

function ref = xd(t)
global m
global g

ref = [sin(0.5*t); cos(0.5*t); m*g*exp(-5*t) + 0.2*sin(0.5*t);
    0.5*cos(0.5*t); -0.5*sin(0.5*t); -5*m*g*exp(-5*t) + 0.1*cos(0.5*t)];
end

function x_dot = ode_file(t, x, xd)
global A
global B
global Kreal

u = Kreal*(xd - x);
x_dot = A*x + B*u;
end