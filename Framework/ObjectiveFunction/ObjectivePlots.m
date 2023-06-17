% Objective function analysis under different mission profiles

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
n = 5; % Trajectory selection n = 1-7

% Initial hover condition
x0 = [0, 0, m*g, 0, 0, 0, 0];

% Drone Linear Dynamics
A = [0*eye(3) eye(3);
    0*eye(3) 0*eye(3)]; % Matrix A

B = [zeros(3,3);
    0 -g 0;
    g 0 0;
    0 0 1/m]; % Matrix B

% Real Weight Matrices
Qreal = diag([1, 1, 1, 5, 5, 5]);
Rreal = 5*eye(3);
Drag = false;

% Solve ARE to obtain the real P and gain K
[Preal, ~, Kreal] = care(A, B, Qreal, Rreal);

% Evaluate the controller and collect data
[t, x] = ode45(@(t,x)ode_file(t, x, xd(t, n), Drag), 0:0.02:180, x0);

% Corrupt data with Gaussian noise
% Large level of noise affects the algorithm
X = x.';
Xd = [];

for i = 1:length(x)
    % Tracking error
    e(:,i) = xd(i*0.02, n) - x(i, 1:6).';

    % Linear Control policy
    Ups(:,i) = Kreal*e(:,i);

    % Real Objective function measurement
    Xi(i) = e(:,i).'*Qreal*e(:,i) + Ups(:,i).'*Rreal*Ups(:,i);

    % Store the desired reference into a matrix
    Xd = [Xd xd(i*0.02, n)];
end

% Show the mean value of the objective function
mean(Xi)

% Visualize the trajectory tracking
figure(1)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
plot3(x(:, 1), x(:,2), x(:, 3), 'r:', 'linewidth', 2);
hold on
plot3(Xd(1, :), Xd(2, :), Xd(3, :), 'b-.', 'linewidth', 2);
grid on;
set(gca, 'fontsize', 16)
xlabel('Pos in $X$ (m)', 'interpreter', 'latex');
ylabel('Pos in $Y$ (m)', 'interpreter', 'latex');
zlabel('Pos in $Z$ (m)', 'interpreter', 'latex');
legend({'Drone tray','Reference'}, 'Interpreter','latex');

% Visualize the objective function
figure(2)
set(gcf,'Position',[100 100 500 200])
set(gcf,'PaperPositionMode','auto')
plot(t, Xi.', 'b', 'linewidth', 2);
grid on;
set(gca, 'fontsize', 16)
xlabel('Time (s)', 'interpreter', 'latex');
ylabel('Objective $\xi$', 'interpreter', 'latex');

function ref = xd(t, n)
global m
global g

if  t <= 30
    z = t/6;
    zp = 1/6;
else
    z = 5;
    zp = 0;
end

switch n
    case 1
        % Fixed point trajectory
        ref = [2; -2; 5; 0; 0; 0];
    case 2
        % Helix trajectory
        ref = [2*cos(0.5*t); 2*sin(0.5*t); z; 
            -0.5*2*sin(0.5*t); 0.5*2*cos(0.5*t); zp];
    case 3
        % Circular trajectory
        ref = [sin(0.5*t); cos(0.5*t); m*g*exp(-5*t) + sin(0.5*t);
            0.5*cos(0.5*t); -0.5*sin(0.5*t);
            -5*m*g*exp(-5*t) + 0.5*cos(0.5*t)];
    case 4
        % Infinity-shape trajectory
        ref = [2*sin(0.125*t); 2*sin(0.25*t); 5-cos(0.5*t);
            2*0.125*cos(0.125*t); 2*0.25*cos(0.25*t); 0.5*sin(0.5*t)];
    case 5
        % Fast Helix trajectory
        ref = [2*cos(5*t); 2*sin(5*t); z; 
            -5*2*sin(5*t); 5*2*cos(5*t); zp];
    case 6
        % Fast circular trajectory
        ref = [sin(5*t); cos(5*t); m*g*exp(-5*t) + 0.2*sin(5*t);
            5*cos(5*t); -5*sin(5*t); -5*m*g*exp(-5*t) + 1*cos(5*t)];
    case 7
        % Fast Infinity-shape trajectory
        ref = [2*sin(0.5*t); 2*sin(t); 5-cos(5*t);
            2*0.5*cos(0.5*t); 2*cos(t); 5*sin(5*t)];
    otherwise
        disp('ERROR: Choose a number between 1-7')
end
end

function var_dot = ode_file(t, var, xd, Drag)
global A
global B
global Kreal

x = var(1:end-1);

Q = diag([1, 1, 1, 5, 5, 5]);
R = 5*eye(3);
if Drag == true
    D = [0*eye(3) 0*eye(3);0*eye(3) -0.5323*eye(3)];
else
    D = 0*eye(6);
end

u = Kreal*(xd - x);

Xi = (xd-x).'*Q*(xd-x) + u.'*R*u;
x_dot = A*x + B*u + D*x;

var_dot = [x_dot; Xi];
end