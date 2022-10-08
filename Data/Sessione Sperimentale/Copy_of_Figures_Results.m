 close all 


%% PLOT FIGURES

% Plot settings
line_tickness = 2; 
color_hat   = [62, 150, 81]/255;        % Torch best slip Prediction
color_GT    = [57,106,177]/255;         % Best Slip GT
color_sig   = [200,36,40]/255;          % Uncertainty
color_slipmu = [107, 76, 154]/255;      % Color for slip,mu graph
color_slip = [0, 102, 204]/255;                     % clor for slip
color_mu = [204, 102, 0]/255;

% font text axis
font_size = 11;
% font_weight = 'normal';
font_weight= 'bold';

% values axis 
font_size_axis = 11;
font_weight_axis= 'normal';
% font_weight_axis= 'bold';

%Segment selection
t_start = 0;
t_end = 5;
ix_start= floor(t_start/Sample_time)+1; 
ix_stop = floor(t_end/Sample_time)+1;


% scale uncertainty 
min_dev = min(standard_dev_Ratio_ML_Torch);
max_dev = max(standard_dev_Ratio_ML_Torch);
min_pred = min(Best_slip_ML_Torch);
max_pred = max(Best_slip_ML_Torch);
standard_dev_Ratio_ML_Torch_norm =  ... 
    ((standard_dev_Ratio_ML_Torch - min_dev) / (max_dev-min_dev) ) * (max_pred -min_pred ) + min_pred ;

%% MED FIGURES

% TITLE
case_exp = 'Wet-Open Loop';
figure(10)
plot(t(ix_start:ix_stop), Best_slip_L_model(ix_start:ix_stop), 'Color', color_GT, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* Ground truth')
hold on
plot(t(ix_start:ix_stop), Best_slip_ML_Torch(ix_start:ix_stop), 'Color',color_hat, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* MLP')
plot(t(ix_start:ix_stop), standard_dev_Ratio_ML_Torch_norm(ix_start:ix_stop), 'Color',color_sig, 'LineWidth', line_tickness-1', 'displayname', 'Uncertainty (Norm)')
grid on
title(['Best slip estimation [Case: ' case_exp  ']'], 'fontweight',font_weight,'fontsize',font_size)
xlabel('Time [s]','fontweight',font_weight,'fontsize',font_size)
ylabel('{\lambda}^* value/norm uncertainty','fontweight',font_weight,'fontsize',font_size)
grid on
legend('show')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)

figure(20)
h(1)=subplot(2,1,1);
plot(t(ix_start:ix_stop), Best_slip_L_model(ix_start:ix_stop), 'Color', color_GT, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* Ground truth')
hold on
plot(t(ix_start:ix_stop), Best_slip_ML_Torch(ix_start:ix_stop), 'Color',color_hat, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* MLP')
grid on
xlabel('Time [s]','fontweight',font_weight,'fontsize',font_size)
ylabel('{\lambda}^* value','fontweight',font_weight,'fontsize',font_size)
legend('show')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


h(2)=subplot(2,1,2);
plot(t(ix_start:ix_stop), standard_dev_Ratio_ML_Torch(ix_start:ix_stop), 'Color',color_sig, 'LineWidth', line_tickness', 'displayname', '{\sigma} Uncertainty')
grid on
xlabel('Time [s]', 'fontweight',font_weight,'fontsize',font_size)
ylabel('{\lambda}^* value', 'fontweight',font_weight,'fontsize',font_size)
legend('show')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)
linkaxes(h,'x')
tupt= suptitle(['Best slip estimation [Case: ' case_exp  ']']);
set(tupt,'FontSize',font_size,'FontWeight',font_weight)


figure(30)
plot(slip_L_model(ix_start:ix_stop), mu_L_model(ix_start:ix_stop), '.', 'Color',color_slipmu, 'LineWidth', line_tickness', 'displayname', '{\lambda} vs {\mu}')
hold on
plot(slip_L_model(ix_start:ix_stop), standard_dev_Ratio_ML_Torch(ix_start:ix_stop), '.', 'Color',color_sig, 'LineWidth', line_tickness-1', 'displayname', '{\lambda} vs Uncertainty')
grid on
title(['Uncertainty({\lambda}) vs ({\lambda}, {\mu}) [Case: ' case_exp  ']'], 'fontweight',font_weight,'fontsize',font_size)
xlabel('{\lambda}')
ylabel('{\mu}')
legend('show')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


figure(40)
h(1)= subplot(3,1,1);
plot(t(ix_start:ix_stop), slip_L_model(ix_start:ix_stop), 'LineWidth', line_tickness', 'displayname', '{\lambda model}')
grid on
xlabel('Time [s]')
ylabel('{\lambda}')
legend('show')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


h(2)= subplot(3,1,2);
plot(t(ix_start:ix_stop), mu_L_model(ix_start:ix_stop), 'LineWidth', line_tickness', 'displayname', '{\mu model}')
grid on
xlabel('Time [s]')
ylabel('{\mu}')
legend('show')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


h(3)= subplot(3,1,3);
plot(t(ix_start:ix_stop), V_model(ix_start:ix_stop), 'LineWidth', line_tickness', 'displayname', 'V speed')
grid on
xlabel('Time [s]')
ylabel('Speed [m/s]')
legend('show')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)

supt= suptitle(['AAAA [Case: ' case_exp  ']']);
set(supt,'FontSize',font_size,'FontWeight',font_weight)

figure(41)
plot(t(ix_start:ix_stop), slip_L_model(ix_start:ix_stop), 'Color',color_slip, 'LineWidth', line_tickness', 'displayname', '{\lambda}')
grid on
hold on
plot(t(ix_start:ix_stop), mu_L_model(ix_start:ix_stop),'Color',color_mu, 'LineWidth', line_tickness', 'displayname', '{\mu}')
xlabel('Time [s]')
ylabel('{\lambda}')
legend('show')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


tit= title(['Trend over time {\lambda} , {\mu} [Case: ' case_exp  ']']);
set(tit,'FontSize',font_size,'FontWeight',font_weight)