close all
clearvars

%% IMPORT TEST FILE 
filename = 'TestKalm.mat'
load(filename)

%% DEFINE STATE SPACE MODEL
%     x_dot=Ax+Bu
%     y=Cx+Du
%
%     A is an Nx-by-Nx real- or complex-valued matrix.
% 
%     B is an Nx-by-Nu real- or complex-valued matrix.
% 
%     C is an Ny-by-Nx real- or complex-valued matrix.
% 
%     D is an Ny-by-Nu real- or complex-valued matrix.

A                       = eye(1)                                ;
B                       = zeros(1)'                             ;
C                       = eye(1)                                ;
D                       = 0                                     ;
I                       = eye(length(C))                        ;

u_k                     = zeros(1)'                             ;
y_k                     = best_slip_Torch_Dropout(:,1)               ;
%% LOAD VARIABLES
k_len                   = length(y_k)                           ;
    
% Init condition
x0 =0.15                                                 ;

% State variables 
x_hat_prior             = 0                                     ;
x_hat_posterior         = x0                                    ;
x_hat_posterior_old     = 0                                     ;

% covariance Matrix
P_prior                 = zeros(1)                              ;        
P_posterior             = zeros(1)                              ;
P_posterior_old         = zeros(1)                              ;

% noises
Q                       = ones(1,k_len).*5e-6                   ;
Q                       = Q.^3                                  ;
    
R                       = (dev_std_Torch_Dropout(:,2).^2).^3     ;
% R                       = ones(1,k_len) * 1.3e-5              ;             


out = ones(1,k_len)                                             ;



%% KALMAN ALGORITHM
for k= 1:k_len 
    if k==1 
        x_hat_posterior_old = x0                                ;                            
    end
    % Prediction Step
    x_hat_prior_new = A*x_hat_posterior_old+B*u_k                 
    P_prior = A*P_posterior_old*A'+ Q(k)                        ;          
    
    % Update Step
    Kgain_k = (P_prior*C')/((C*P_prior*C')+R(k))                ;                   
    x_hat_posterior =  ... 
        x_hat_prior_new + Kgain_k*(y_k(k)-C*x_hat_prior_new)    ;
    P_posterior = (I-Kgain_k*C)*P_prior                         ;                           
    
    P_posterior_old = P_posterior                               ;
    x_hat_posterior_old = x_hat_posterior                       ;
    out(k)= x_hat_posterior                                     ;
    
end

t_old = t1-1;
[~, idx_low]= min(abs(t_old));

t_old2 = t1-2;
[~, idx_up]= min(abs(t_old2));

standard_var_ML_Torch = dev_std_Torch_Dropout(:,2).^2;
mean_val = mean(standard_var_ML_Torch(idx_low:idx_up,1))

figure
p1= plot(t1,best_slip_Torch_Dropout(:,1))
hold on
p2 = plot(t1, out)
grid on
xlabel('Simulation Time [s]')
ylabel('{\lambda}^* Estimated')
title(['{\lambda}^* Kalman Filtering',])
legend([p1, p2],'{\lambda}^* PytorchModel', '{\lambda}^* PytorchModel Filtered')
