% Script Gradient Inverse Reinforcement Learning
% This script obtains the objective function of optimal LQR policies
% Drone dynamics assumed to be linear

clc
close all
clear all

% Seed to obtain the same results
rng(800)

% Drone parameters
g = 9.81;
m = 0.467;

% Drone  linear matrices
A = [0*eye(3) eye(3);0*eye(3) 0*eye(3)];
B = [0*eye(3);0 -g 0;g 0 0;0 0 1/m];

% Real Weight Matrices
% Q0 = randn(6,6); 
% Qreal1 = 1/2*(Q0.'*Q0);
Qreal = diag([1, 1, 1, 5, 5, 5]);
Rreal = 5*eye(3);

% Initial weight matrix (We use S instead of Q)
S0 = 0.1*eye(6);

% Real optimal control solution
[Preal, ~, Kreal] = care(A, B, Qreal, Rreal);

imax = 5000;   % Max number of episodes
a = 0.1;       % Learning rate
S = [];        % Store values of the weight matrix
PP = [];       % Store values of the kernel matrix P
KK = [];       % Store values of the control gain K
Snorm = [];    % Store RMSSNE of S(i+1)-S(i)
Knorm = [];    % Store RMSSNE of K(i+1)-Kreal
Pnorm = [];    % Store RMSSNE of P(i+1)-P(i)
S_realnorm = []; % Store RMSSNE of S(i)-Qreal
P_realnorm = []; % Store RMSSNE of P(i)-Preal
Rhat = 5*eye(3);    % Weight matrix Rhat

% Gradient IRL loop
for i = 1:imax
    % Stabilizing control policy
    [P, ~, K] = care(A, B, S0, Rhat);

    % Store control gain K
    KK = [KK K];

    % Control gain error
    e = K-Kreal;

    % Norm of the control gain error
    Knorm(i) = norm(e);

    % Estimate a new kernel matrix P2
    P2 = P - a*(B*inv(Rhat)*e + e.'*inv(Rhat)*B.');
    
    % Store weight and kernel matrices
    S = [S S0];
    PP = [PP P2];

    % Vectorize the elements of S0
    Theta(:,i) = reshape(S0,[],1);

    % Compute improved weight matrix S(i+1)
    S1 =-(A.'*P2 + P2*A - P2*B*inv(Rhat)*B.'*P2);

    % Compute spectral norm of S and P
    Snorm(i) = norm(S1-S0);
    S_realnorm(i) = norm(Qreal - S0);
    P_realnorm(i) = norm(Preal - P);
    Pnorm(i) = norm(P2 - P);

    % Set weight matrix S0 as S1 and return to the loop
    S0 = S1;
end

% Final weight matrix Q
Qreal
Q = reshape(Theta(:,end), 6 , 6)

% Final control gain
Kreal
K

% To obtain the RMSSNE of K(i+1)-K(i)
K1 = KK(:,1:6);
j = 1;
for i=1:length(K):length(KK)-6
    K2 = KK(:,i+6:i+11);
    % RMSSNE K(i+1)-K(i)
    Kinorm(j) = norm(K1-K2);
    K1 = K2;
    j = j + 1;
end

% Plots
figure(1)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
t = 0:imax-1;
hold on
plot(t(:),Knorm(:),'linewidth',2);
plot(t(:),Pnorm(:),'linewidth',2);
plot(t(:),Snorm(:),'linewidth',2);
grid on;
set(gca,'fontsize',16);
ylabel({'$\|\cdot\|$'},'interpreter','latex');
xlabel({'Episode $i$'},'interpreter','latex');
legend({'$\|\bf{\mathcal{K}}_{i+1}-\bf{\mathcal{K}}_i\|$',...
    '$\|\bf{P}_{i+1}-\bf{P}_i\|$', ...
    '$\|\bf{Q}_{i+1}-\bf{Q}_i\|$'},'interpreter','latex');

j = linspace(0,imax-1,imax-1);
figure(2)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
hold on;
plot(j,Theta([1:6,8:12,15:18, 22:24, 29:30, 36],1:length(j)),'linewidth',2)
set(gca,'fontsize',16);
grid on;
xlabel({'Episodes $i$'},'interpreter','latex');
ylabel({'$\Theta_i$ values'},'interpreter','latex');