close all
clearvars

%% IMPORT TEST FILE 
%  filename = 'OL_DWS_pythorc_stddev.mat'                          ;
filename = 'TestKalm.mat'                          ;
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
t1                      = [0:1:k_len-1].*Sample_time            ;
time = t1                                                       ; 
% Init condition
x0 =0.15                                                        ;
kalman_out = ones(1,k_len)                                      ;

% State variables 
x_hat_prior             = 0                                     ;
x_hat_posterior         = x0                                    ;
x_hat_posterior_old     = 0                                     ;

% covariance Matrix
P_prior                 = zeros(1)                              ;        
P_posterior             = zeros(1)                              ;
P_posterior_old         = zeros(1)                              ;

% noises parameters
Q_start                 = ones(1,k_len)                         ;
R_start                 = dev_std_Torch_Dropout(:,2)            ;
% R                       = ones(1,k_len) * 1.3e-5              ;
% R                       = ones(1,k_len) * 1.3e-13
kg_num = zeros(1,k_len);
kg_den = zeros(1,k_len);
k_gain = zeros(1,k_len);

%% MAGIC
%get assumption of R-order: 
t_old = t1-1;
[~, idx_low]= min(abs(t_old));
t_old2 = t1-2;
[~, idx_up]= min(abs(t_old2));
standard_var_ML_Torch = dev_std_Torch_Dropout(:,2).^2;
mean_val = mean(standard_var_ML_Torch(idx_low:idx_up,1));
power=ceil(log10(mean_val)-1);
power_ref = ones(1,k_len).*10^(power);
residual= abs(R_start-power_ref');
new_R = (residual).*10^(-power);
figure
plot(new_R)

% noises
Q                       = ones(1,k_len).*5e-6                   ;
Q                       = Q.^3                                  ;
    
R                       = (dev_std_Torch_Dropout(:,2).^2).^3     ;
% R                       = ones(1,k_len) * 1.3e-5              ;    


%% KALMAN ALGORITHM
for k= 1:k_len 
    if k==1 
        x_hat_posterior_old = x0                                ;                            
    end
    % Prediction Step
    x_hat_prior_new = A*x_hat_posterior_old+B*u_k               ;           
    P_prior = A*P_posterior_old*A'+ Q(k)                        ;          
    
    % Update Step
    kg_num(k) = (P_prior*C')                                    ;
    kg_den(k) =((C*P_prior*C')+R(k))                            ;
    k_gain(k) =  kg_num(k)/kg_den(k)                            ;                   
    x_hat_posterior =  ... 
        x_hat_prior_new + k_gain(k)*(y_k(k)-C*x_hat_prior_new)  ;
    P_posterior = (I-k_gain(k)*C)*P_prior                       ;                           
    
    P_posterior_old = P_posterior                               ;
    x_hat_posterior_old = x_hat_posterior                       ;
    kalman_out(k)= x_hat_posterior                              ;
    
end


%% PLOTS


% kalman filter output
figure
p1= plot(time,best_slip_Torch_Dropout(:,1))
hold on
p2 = plot(time, kalman_out)
grid on
p3 = plot(t,MaxT(:,1))
xlabel('Simulation Time [s]')
ylabel('{\lambda}^* Estimated')
title('{\lambda}^* Kalman Filtering')
legend([p1, p2, p3],'{\lambda}^* PytorchModel', '{\lambda}^* PytorchModel Filtered', '{\lambda} GT')



% Kalman filter Gain : numerator, denominator, num/den
% 
figure
subplot(3,1,1)
p1 = plot(time, kg_num)
xlabel('time [s]')
ylabel('value')
title('Numerator')
grid on
subplot(3,1,2)
p1 = plot(time, kg_den)
xlabel('time [s]')
ylabel('value')
title('Denumerator')
grid on
subplot(3,1,3)
p1 = plot(time, k_gain)
xlabel('time [s]')
ylabel('K gain')
title('K gain')
grid on
suptitle('Kalman filter Gain')


% Kalman filter: Q,R : 
figure
subplot(2,1,1)
p1 = plot(time, Q)
xlabel('time [s]')
ylabel('value')
title('Q')
grid on
subplot(2,1,2)
p1 = plot(time, R)
xlabel('time [s]')
ylabel('value')
title('R')
grid on
suptitle('Kalman filter noises')




