% Script Unrestricted Inverse Reinforcement Learning
% This script obtains the objective function of optimal LQR policies
% Drone dynamics assumed to be linear

clc
close all
clear all

% Seed to obtain the same results
rng(800)

% Parameters of the drone
g = 9.81;
m = 0.467;

% Drone Linear Dynamics
A = [0*eye(3) eye(3);
    0*eye(3) 0*eye(3)]; % Matrix A

B = [zeros(3,3);
    0 -g 0;
    g 0 0;
    0 0 1/m]; % Matrix B

% Real Weight Matrices
%Q0 = randn(6,6); 
%Qreal = 1/2*(Q0.'*Q0);
%Q0 = 0*eye(6);
Qreal = diag([1, 1, 1, 5, 5, 5]);
Rreal = 5*eye(3);

% Solve ARE to obtain the real P and gain K
[Preal, ~, Kreal] = care(A, B, Qreal, Rreal);


% Initialize random weight matrices for the Model-Based IRL
Q(:, :, 1) = Qreal/2;
Rest = 5*eye(3);
R0 = 5*eye(3);
Q0 = Qreal/2;
% Find an initial stabilizable control policy
[~, ~, K(:, :, 1)] = care(A, B, Q0, R0);

n = length(A);
episodes = 5000;

% Main Loop of the model-based IRL
for i = 1: episodes
    % Solve a Least-squared Lyapunov Recursion model
    W = (K(:, :, i) - Kreal).'*Rest*(K(:, :, i) - Kreal) + ...
        Q(:, :, i) + K(:, :, i).'*Rest*K(:, :, i);
    vecW = reshape(-W, [], 1);
    vecP = pinv(kron(eye(n), (A - B*K(:, :, i))).' + ...
        kron((A - B*K(:, :, i)), eye(n)).')*vecW;

    % New kernel matrix P(i)
    P(:, :, i) = reshape(vecP, n, n);

    % New improved control gain K(i+1)
    K(:, :, i + 1) = Rest\B.'*P(:, :, i);  
    
    % Build matrices for the IRL for Drone 2
    Q(:, :, i + 1) = -A.'*P(:, :, i) - P(:, :, i)*A + ...
                K(:, :, i + 1).'*Rest*K(:, :, i + 1);   
   
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

% RMSSNE of K, P and Q
for i = 1 : size(P,3) - 1
    episode(i) = i-1;
    Pnorm(i) = norm(P(:, :, i+1) - P(:, :, i));
    Qnorm(i) = norm(Q(:, :, i+1) - Q(:, :, i));
    Knorm(i) = norm(K(:, :, i+1) - K(:, :, i));
    Preal_norm(i) = norm(P(:, :, i) - Preal);
    Qreal_norm(i) = norm(Q(:, :, i) - Qreal);
    Kreal_norm(i) = norm(K(:, :, i) - Kreal);
end

%Plots
figure(1)
set(gcf,'Position',[100 100 500 250])
set(gcf,'PaperPositionMode','auto')
plot(episode(1:21), Knorm(1:21),'>-', 'linewidth', 2);
hold on
plot(episode(1:21), Pnorm(1:21), 's-', 'linewidth', 2);
plot(episode(1:21), Kreal_norm(1:21),'>-', 'linewidth', 2);
plot(episode(1:21), Preal_norm(1:21), 's-', 'linewidth', 2);
set(gca, 'fontsize', 16)
grid on
xlabel({'Episodes $i$'},'interpreter','latex');
ylabel({'$\|\cdot\|$'},'interpreter','latex');
legend({'$\|\textbf{K}_{i+1}-\textbf{K}_i\|$', ...
    '$\|\textbf{P}_{i+1}-\textbf{P}_i\|$', ...
    '$\|\textbf{K}_{i}-\textbf{K}\|$', ...
    '$\|\textbf{P}_{i}-\textbf{P}\|$', ...
    },'interpreter','latex');

figure(2)
set(gcf,'Position',[100 100 500 250])
set(gcf,'PaperPositionMode','auto')
hold on
plot(episode(1:21), Qnorm(1:21), 'o-', 'linewidth', 2);
plot(episode(1:21), Qreal_norm(1:21), 'o-', 'linewidth', 2);
set(gca, 'fontsize', 16)
grid on
xlabel({'Episodes $i$'},'interpreter','latex');
ylabel({'$\|\cdot\|$'},'interpreter','latex');
legend({
    '$\|\textbf{Q}_{i+1}-\textbf{Q}_i\|$', ...
    '$\|\textbf{Q}_{i}-\textbf{Q}\|$', ...
    },'interpreter','latex');