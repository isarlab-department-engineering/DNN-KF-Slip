 close all 


%% PLOT FIGURES

% Plot settings
line_tickness = 2; 
color_hat   = [62, 150, 81]/255;        % Torch best slip Prediction
color_GT    = [57,106,177]/255;         % Best Slip GT
color_sig   = [200,36,40]/255;          % Uncertainty

color_kal_out = [225, 151, 76]/255;     % Filtered prediction
color_slipmu = [107, 76, 154]/255;      % Color for slip,mu graph

% font text axis
font_size = 12;
% font_weight = 'normal';
font_weight= 'bold';

% values axis 
font_size_axis = 11;
% font_weight_axis= 'normal';
font_weight_axis= 'bold';



% scale uncertainty 
min_dev = min(standard_dev_Ratio_ML_Torch);
max_dev = max(standard_dev_Ratio_ML_Torch);
min_pred = min(Best_slip_ML_Torch);
max_pred = max(Best_slip_ML_Torch);
standard_dev_Ratio_ML_Torch_norm =  ... 
    ((standard_dev_Ratio_ML_Torch - min_dev) / (max_dev-min_dev) ) * (max_pred -min_pred ) + min_pred ;

%% MED FIGURES

figure(10)
% title('Best slip estimation', 'fontweight',font_weight,'fontsize',font_size)
plot(t, Best_slip_L_model, 'Color', color_GT, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* MLP')
hold on
plot(t, Best_slip_ML_Torch, 'Color',color_hat, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* GT')
plot(t, standard_dev_Ratio_ML_Torch_norm, 'Color',color_sig, 'LineWidth', line_tickness-1', 'displayname', '{\sigma} Uncertainty')
grid on
legend()
xlabel('Time [s]','fontweight',font_weight,'fontsize',font_size)
ylabel('{\lambda}^* value','fontweight',font_weight,'fontsize',font_size)
grid on
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


figure(20)
% suptitle('Best slip estimation', 'fontweight',font_weight,'fontsize',font_size)
h(1)=subplot(2,1,1)
plot(t, Best_slip_L_model, 'Color', color_GT, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* MLP')
hold on
plot(t, Best_slip_ML_Torch, 'Color',color_hat, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* GT')
grid on
legend()
xlabel('Time [s]','fontweight',font_weight,'fontsize',font_size)
ylabel('{\lambda}^* value','fontweight',font_weight,'fontsize',font_size)
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


h(2)=subplot(2,1,2)
plot(t, standard_dev_Ratio_ML_Torch, 'Color',color_sig, 'LineWidth', line_tickness', 'displayname', '{\sigma} Uncertainty')
grid on
legend()
xlabel('Time [s]', 'fontweight',font_weight,'fontsize',font_size)
ylabel('{\lambda}^* value', 'fontweight',font_weight,'fontsize',font_size)
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)
linkaxes(h,'x')



stdd = standard_dev_Ratio_ML_Torch.*Best_slip_ML_Torch;


figure(21)
% suptitle('Best slip estimation', 'fontweight',font_weight,'fontsize',font_size)
h(1)=subplot(2,1,1)
plot(t, Best_slip_L_model, 'Color', color_GT, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* MLP')
hold on
plot(t, Best_slip_ML_Torch, 'Color',color_hat, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* GT')
plot(t, Best_slip_ML_Torch+3*stdd, 'Color',color_hat, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* GT')
plot(t, Best_slip_ML_Torch-3*stdd, 'Color',color_hat, 'LineWidth', line_tickness', 'displayname', '{\lambda}^* GT')
grid on
legend()
xlabel('Time [s]','fontweight',font_weight,'fontsize',font_size)
ylabel('{\lambda}^* value','fontweight',font_weight,'fontsize',font_size)
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)
axis([0, 10, 0, 0.3])


h(2)=subplot(2,1,2)
plot(t, stdd, 'Color',color_sig, 'LineWidth', line_tickness', 'displayname', '{\sigma} Uncertainty')
grid on
legend()
xlabel('Time [s]', 'fontweight',font_weight,'fontsize',font_size)
ylabel('{\lambda}^* value', 'fontweight',font_weight,'fontsize',font_size)
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)
linkaxes(h,'x')





figure(30)
plot(slip_L_model, mu_L_model, '.', 'Color',color_slipmu, 'LineWidth', line_tickness', 'displayname', '({\sigma}, {\mu})')
hold on
plot(slip_L_model, standard_dev_Ratio_ML_Torch*3, '.', 'Color',color_sig, 'LineWidth', line_tickness-1', 'displayname', '{\sigma} Uncertainty')
grid on
legend()
xlabel('{\lambda}')
ylabel('{\mu}')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


figure(40)
h(1)= subplot(3,1,1)
plot(t, slip_L_model, 'LineWidth', line_tickness', 'displayname', '{\lambda model}')
grid on
legend()
xlabel('Time [s]')
ylabel('{\lambda}')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


h(2)= subplot(3,1,2)
plot(t, mu_L_model, 'LineWidth', line_tickness', 'displayname', '{\mu model}')
grid on
legend()
xlabel('Time [s]')
ylabel('{\mu}')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)


h(3)= subplot(3,1,3)
plot(t, V_model, 'LineWidth', line_tickness', 'displayname', 'V speed')
grid on
legend()
xlabel('Time [s]')
ylabel('V speed [m/s]')
set(gca,'FontSize',font_size_axis,'fontweight',font_weight_axis)
linkaxes(h,'x')

clearvars


