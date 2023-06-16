%% Off-line Reservoir Computing %
% This script runs the RC method using off-line data trajectories
% Input X which contains an N x T matrix
% N is the number of features
% T is the length of the time-sequences

clc
close all
clear all

% Global variables 
global A
global Win
global units
global Wout
global w
global B
global ft
global T

% Seed to obtain the same results
rng(800);

% Prediction time window
ft = 100;   % Number of steps for prediction ft<length(X)
T = 0.02;    % Sampling time

% Load the data
load('Drone1/X_tray.mat', 'X');

% Add some noise to the trajectories to model sensor noise
Xnoise = X + 1e-1*std(X,[],'all').*randn(size(X));

% Split into training and testing data
X_train = Xnoise(1:3, 1:(end - 1)/2 - ft); % Training data
Y_train = Xnoise(1:3, 1 + ft:(end - 1)/2); % Training prediction data
X_test = Xnoise(1:3, length(X_train) + 1:end - ft); % Testing data
Y_test = Xnoise(1:3, length(X_train) + 1 + ft: end); % Testing prediction data

% Parameters of the Reservoir network
units = 30;  % Number of units
A0 = randn([units, units]); % Random generated matrix
A = - 0.5*(A0.'*A0); % Construct a Negative-definite matrix
Win = randn([units, 3]); % Random input weights

% Initial conditions
r0 = zeros(1, units);  % Reservoir states initial condition
t0 = 0;  % Initial time of ODE
tf = T;  % Final time of ODE

R = [r0.';1]; % Store the reservoir states in each time step
time = [t0];  % Store time steps

% Main Loop of the Standard RC network
for i = 1: length(X_train)-1
    % Solve the ODE in a time interval [t0 tf]
    [t,r] = ode23(@(t,r)ode_file(t,r, X_train(:, i)),[t0 tf] , r0);
    
    % Obtain the final reservoir states and time of iteration i 
    r1= r(size(r,1), :);
    t1 = t(length(t));

    % Store the reservoir states and time
    R = [R [r1.';1]];
    time = [time t1];

    % Update initial and final times, and initial reservoir states
    t0 = tf;
    tf = tf + T;
    r0 = r1;
end

% Obtain the decoder weights using a regularised Least squares
lambda = 0.5; % Regularisation term
%Wdec = Y_train*pinv(R); % Without regularisation

% Decoder weights
Wdec = Y_train*pinv(R.'*R + lambda*eye(length(X_train)))*R.'; 
Wout = Wdec(:,1:end-1);
w = Wdec(:,end);

% Reservoir Prediction
y = Wdec*R;

% Reservoir Computing with SVM Nonlinear Regression model (1 x dimension)
svmrc1 = fitrsvm(R(1:end-1, :).', Y_train(1,:).','KernelFunction','rbf');
svmrc2 = fitrsvm(R(1:end-1, :).', Y_train(2,:).','KernelFunction','rbf');
svmrc3 = fitrsvm(R(1:end-1, :).', Y_train(3,:).','KernelFunction','rbf');

ysvm1_train = predict(svmrc1, R(1:end-1, :).');
ysvm2_train = predict(svmrc2, R(1:end-1, :).');
ysvm3_train = predict(svmrc3, R(1:end-1, :).');

YSVM_Train = [ysvm1_train, ysvm2_train, ysvm3_train].';

% Reservoir Computing with MLP Regression model (1 x dimension)
mlp1 = fitrnet(R(1:end-1, :).', Y_train(1,:).',"LayerSizes", [10 30 10]);
mlp2 = fitrnet(R(1:end-1, :).', Y_train(2,:).',"LayerSizes", [10 30 10]);
mlp3 = fitrnet(R(1:end-1, :).', Y_train(3,:).',"LayerSizes", [10 30 10]);

ymlp1_train = predict(mlp1, R(1:end-1, :).');
ymlp2_train = predict(mlp2, R(1:end-1, :).');
ymlp3_train = predict(mlp3, R(1:end-1, :).');

YMLP_Train = [ymlp1_train, ymlp2_train, ymlp3_train].';

%% Physics Informed Reservoir Computing %%

B = zeros(units, units); % Physics Informed weights
r0 = zeros(1, units);    % Initial reservoir states
t0 = 0;                  % Initial time of the ODE
tf = T;                  % Final time of the ODE

R2 = [r0.';1];           % Store the reservoir states in each time step

for i = 1: length(X_train)-1
    % Solve the ODE in a time [t0 tf]
    [t,r] = ode23(@(t,r)ode_file2(t,r, X_train(:, i), Y_train(:,i)), ...
        [t0 tf] , r0);

    % Obtain the final reservoir states of iteration i 
    r1= r(size(r,1),:);

    % Store the reservoir states 
    R2 = [R2 [r1.';1]];

    % Updates times and initial reservoir states
    t0 = tf;
    tf = tf + T;
    r0 = r1;
end

% Physics Informed Prediction
y_est = Wdec*R2;

%Test the results in Test Data
r0RC = (pinv(Wout)*(Y_test(:,1)-w)).';
r0PIRC = (pinv(Wout)*(Y_test(:,1)-w)).';
t0 = 0;
tf = T;
time2 = [t0];
YRC = [Y_test(:,1)];
YPIRC = [Y_test(:,1)];
RRC = [r0RC.'];
for i = 1: length(X_test)-1
    % Solve the ODE in a time [t0 tf]
    [tRC, rRC] = ode23(@(tRC,rRC)ode_file(tRC,rRC, X_test(:, i)), ...
        [t0 tf], r0RC);
    [tPIRC,rPIRC] = ode23(@(tPIRC,rPIRC)ode_file3(tPIRC,rPIRC, ...
        X_test(:, i)), [t0 tf] , r0PIRC);

    % Obtain the final reservoir states of iteration i 
    r1RC= rRC(size(rRC,1),:);
    r1PIRC= rPIRC(size(rPIRC,1),:);
    t1 = tPIRC(length(tPIRC)); 
    yRC = Wout*r1RC.' + w;
    yPIRC = Wout*r1PIRC.' + w;

    % Store the reservoir states 
    YRC = [YRC yRC];
    RRC = [RRC r1RC.'];
    time2 = [time2 t1];
    YPIRC = [YPIRC yPIRC];

    % Updates times and initial reservoir states
    t0 = tf;
    tf = tf + T;
    r0RC = r1RC;
    r0PIRC = r1PIRC;
end

ysvm1_test = predict(svmrc1, RRC.');
ysvm2_test = predict(svmrc2, RRC.');
ysvm3_test = predict(svmrc3, RRC.');

YSVM_Test = [ysvm1_test ysvm2_test ysvm3_test].';

ymlp1_test = predict(mlp1, RRC.');
ymlp2_test = predict(mlp2, RRC.');
ymlp3_test = predict(mlp3, RRC.');

YMLP_Test = [ymlp1_test, ymlp2_test, ymlp3_test].';

MSESVM_TRAIN =  mean(mean((Y_train(1:3,:) - YSVM_Train).^2))
MSESVM_TEST =  mean(mean((Y_test(1:3,:) - YSVM_Test).^2))
MSEMLP_TRAIN =  mean(mean((Y_train(1:3,:) - YMLP_Train).^2))
MSEMLP_TEST =  mean(mean((Y_test(1:3,:) - YMLP_Test).^2))
MSERC_TRAIN = mean(mean((Y_train(1:3,:) - y).^2))
MSEPIRC_TRAIN = mean(mean((Y_train(1:3,:)-y_est).^2))
MSERC_TEST = mean(mean((Y_test(1:3,:) - YRC).^2))
MSEPIRC_TEST = mean(mean((Y_test(1:3,:)-YPIRC).^2))

%% Plots of the results
% 3D Trajectory
figure(2)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
c1 = 0.5*[1 1 1]; c2 = 0.7*[1 1 1];
plot3(Y_test(1,:), Y_test(2,:), Y_test(3,:), 'LineWidth', 2, ...
    'Color', c2);
hold on;
plot3(YRC(1,:), YRC(2,:), YRC(3,:),'r-o', 'LineWidth', 2, ...
    'MarkerSize', 0.5);
grid on;
plot3(YPIRC(1,:), YPIRC(2,:), YPIRC(3,:),'m:', 'LineWidth', 2, ...
     'MarkerSize', 0.5);
xlabel({'Position in $x$ (m)'}, 'interpreter', 'latex');
ylabel({'Position in $y$ (m)'}, 'interpreter', 'latex');
zlabel({'Position in $z$ (m)'}, 'interpreter', 'latex');
set(gca,'fontsize', 14);
legend({'Ground truth', 'RC Prediction', 'PIRC Prediction'}, ...
     'interpreter','latex');
%axis([-4 0 -2 2 0.5 1.25]);

% Position in X
figure(3)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
plot(time2, Y_test(1,:),'linewidth', 2, 'Color', c2);
hold on;
plot(time2, YRC(1,:), 'r-.', 'LineWidth', 2)
plot(time2, YPIRC(1,:), 'm:', 'LineWidth', 2)
grid on
xlabel({'Time (s)'}, 'interpreter', 'latex');
ylabel({'Position in $X$ (m)'}, 'interpreter', 'latex');
legend({'Ground truth', 'RC Prediction', 'PIRC Prediction'}, ...
     'interpreter','latex');
set(gca,'fontsize', 14);

% Position in Y
figure(4)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
plot(time2, Y_test(2,:), 'linewidth', 2, 'Color', c2);
hold on;
plot(time2, YRC(2,:), 'r-.', 'LineWidth', 2)
plot(time2, YPIRC(2,:), 'm:', 'LineWidth', 2)
grid on
xlabel({'Time (s)'}, 'interpreter', 'latex');
ylabel({'Position in $Y$ (m)'}, 'interpreter', 'latex');
legend({'Ground truth', 'RC Prediction', 'PIRC Prediction'}, ...
     'interpreter','latex');
set(gca,'fontsize', 14);

% Position in Z
figure(5)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
plot(time2, Y_test(3,:), 'linewidth', 2, 'Color', c2);
hold on;
plot(time2, YRC(3,:), 'r-.', 'LineWidth', 2)
plot(time2, YPIRC(3,:), 'm:', 'LineWidth', 2)
grid on
xlabel({'Time (s)'}, 'interpreter', 'latex');
ylabel({'Position in $Z$ (m)'}, 'interpreter', 'latex');
legend({'Ground truth', 'RC Prediction', 'PIRC Prediction'}, ...
     'interpreter','latex');
set(gca,'fontsize', 14);

%% Reservoir Computing ODEs

function r_dot = ode_file(t, r, x)
%Standard Reservoir Computing Network

% global variables
global A
global Win

% Reservoir dynamics
r_dot = tanh(A*r + Win*x);
end

function r_dot = ode_file2(t, r, x, y)
% Physics informed reservoir computing ODE

% global variables
global A
global Win
global units
global Wout
global w
global B
global ft

% Reservoir states error
r_tilde = r - pinv(Wout)*(y - w);

% Taylor series expansion term
Dsig = diag(sech(Win*x+(A + B)*r).^2);

% Definition of the error dynamics
fx = -Wout*Dsig*A*r_tilde;
gx = -kron(Wout*Dsig,r_tilde.');

% Prediction error
e =Wout*r + w - y;

% Feedback linearisation controller
u = - pinv(gx)*(fx + 0.5*e);
B = reshape(u, units, units); % Control reshape into weights
eps = 0.0001*ft; % factor to increase the eigenvalues of matrix B
B = -1/2*(B.'*B + eps*eye(units));

% Reservoir dynamics
r_dot = tanh((A + B)*r + Win*x);
end

function r_dot = ode_file3(t, r, x)
% Physics informed reservoir computing ODE

% global variables
global A
global Win
global B

% Reservoir dynamics
r_dot = tanh((A + B)*r + Win*x);
end
