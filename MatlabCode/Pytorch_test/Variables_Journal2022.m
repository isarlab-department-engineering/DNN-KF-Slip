% File per il set-up delle variabili utilizzate nella simulazione;

%% Dati Fisici relativi allo scenario simulativo

%% UNIPG
Numruote                = 2;
Massa_drone_nominale    = 1600;    
Massa_drone_eq          = Massa_drone_nominale / Numruote;      % (Kg)
Massa_ruota             = 10;                                   % (Kg)
Raggio_pneumatico       = 0.3;                                  % (m)
Accelerazione_grav      = 9.8;                                  % (m/sec^2)
Momento_inerzia         = (0.5 * Massa_ruota *...
    Raggio_pneumatico^2);                                       % (Kg * m^2)
Fz_routa = Massa_drone_eq * Accelerazione_grav;                    % (N)


%% Condizioni iniziali modello
v0=80;
omega0=v0/Raggio_pneumatico;

%% Condizioni finale simulazione
v_min_stop = 15; % (m/s)

%% Dati per la simulazione
% Sample_time             = 5*10^-3;    % (sec)
ML_sample_time          = 5*10^-3;    % (sec)
Sample_time = ML_sample_time ;
%% Potenze Rumore Bianco
Pow_WN_model            = 10^-7;    % (Power Spectral Density) 
Pow_WN_vout_model       = 10^-7;    % (Power Spectral Density) 
Pow_WN_wout_model       = 10^-7;    % (Power Spectral Density
%% Valori lambda(slip) da 0->1 precalcolato
num_samples_lambda = 5000;
lambda_dummy = linspace(0, 1, num_samples_lambda);
    

%% Parametri Blocco Recursive Least Squares Estimator 

%Initial Estimate -> Internal
RLS_Init_param_values   = [1.22 -0.45 0.18 -1.19 -0.25]/2; 
RLS_Ro_Covar_Mat    = 2*10^1;
RLS_Forgetting_factor       = 0.999;

% Guadagno funzione LP (linear approximating per Burckhardt (R. de Castro)
G1= -4.99;
G2= -18.43;
G3= -65.62;


%% Blocco derivativo stima mu

% v_meas --> |FS| ---> Vdot

% State-space model:
% dx/dt = Ax + Bu
%  y = Cx + Du

num_tf_v = [1 0];
den_tf_v = [0.01 1];
[Av,Bv,Cv,Dv] = tf2ss(num_tf_v,conv(den_tf_v,den_tf_v));

v_state_space_A = Av;
v_state_space_B = Bv;
v_state_space_C = [Cv; eye(2)];
v_state_space_D = [Dv; 0;0]
v_state_space_init = [inv(Av)*-Bv*v0];

% w_meas --> |FS| ---> wdot
num_tf_w = [1 0];
den_tf_w = [0.01 1];
[Aw,Bw,Cw,Dw] = tf2ss(num_tf_w,conv(den_tf_w,den_tf_w));
w_state_Space_A = Aw;
w_state_Space_B = Bw;
w_state_Space_C = [Cw; eye(2)];
w_state_Space_D = [Dw; 0;0] ;
w_state_space_init = [inv(Aw)*-Bw*omega0];

%% Blocco Controllo SMC + CI 

% epsilon: s0 + eps , s0 - eps
SMC_epsilon = 0.001; 
%slack variable for extra margin to the upper bound error
SMC_beta_zero = 0.001;
% gain for convrgence in the slididng surface
SMC_kappa = 10; 

%% Blocco Controllo PI
%default
PI_kp = 0.5;
den_Tdel = [0 0.2]
PI_ki = 20



