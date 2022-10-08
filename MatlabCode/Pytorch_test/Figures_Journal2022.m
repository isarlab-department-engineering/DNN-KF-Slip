% close all 
% clearvars
 close all 


%Coppia frenante richiesta
figure(10)
plot(t,Torque_req)
xlabel('time')
ylabel('Newton')
title('Coppia frenante richiesta')
grid on

%Velocita' veicolo e ruota
figure(20)
suptitle('Vehicle  and Wheel Speed')
plot(t, speed(:,1))
xlabel('time')
ylabel('m/sec')
hold on
plot(t,speed(:,2).*Raggio_pneumatico)
grid on
legend('v', 'r{\omega}')


figure(30)
subplot(2,1,1)
suptitle('valori {\mu},{\lambda}')
plot(lambda_model, mu_model,'.')
hold on
plot(lambda_model, mu_model,'.')
xlabel('{\lambda}')
ylabel('{\mu}')
grid on
legend('no noise', 'noise')
title('Valori generati dal modello')
subplot(2,1,2)
plot(lambda_hat, mu_hat,'.')
xlabel('{\lambda}')
ylabel('{\mu}')
grid on
title('Valori collezionati')


%lambda e mu nel tempo
figure(40)
subplot(2,1,1)
suptitle('Andamento {\lambda},{\mu} nel tempo')
plot(t,lambda_model)
hold on
plot(t, lambda_hat)
grid on
xlabel('Time (sec)')
ylabel('{\lambda}')
title('{\lambda}')
legend('{\lambda} model', '{\lambda} calculated')
subplot(2,1,2)
plot(t, mu_model)
hold on
plot(t, mu_hat)
grid on
xlabel('Time (sec)')
ylabel('{\mu}')
title('{\mu}')
legend('{\mu} model', '{\mu} calculated')



figure(50)
subplot(4,1,1)
p1 = plot(t1,best_slip_GT(:,1),'.','DisplayName', '{\lambda}^* Model');
hold on
p2_1 = plot(t1, best_slip_SKlearn(:,1),'.', 'DisplayName', '{\lambda}^* SKLearn');
grid on
xlabel('Time (sec)')
ylabel('best slip {\lambda}')
legend([p1,p2_1])
title('Predizione Rete SKLearn')
set(gca, 'ylim', [0, 1]);

subplot(4,1,2)
p1 = plot(t1,best_slip_GT(:,1), '.','DisplayName', '{\lambda}^* Model');
hold on
p2_1 = plot(t1, best_slip_Torch_NoDropout(:,1),'.', 'DisplayName', '{\lambda}^* Pytorch NoDrop');
grid on
xlabel('Time (sec)')
ylabel('best slip {\lambda}')
legend([p1,p2_1])
title('Predizione Rete PyTorch senza Dropout')
set(gca, 'ylim', [0, 1]);

subplot(4,1,3)
p1 = plot(t1,best_slip_GT(:,1), '.','DisplayName', '{\lambda}^* Model');
hold on
p2_1 = plot(t1, best_slip_Torch_Dropout(:,1),'.', 'DisplayName', '{\lambda}^* Pytorch NoDrop');
grid on
xlabel('Time (sec)')
ylabel('best slip {\lambda}')
legend([p1,p2_1])
title('Predizione Rete PyTorch con Dropout')
set(gca, 'ylim', [0, 1]);

subplot(4,1,4)
p1 = plot(t1,best_slip_GT(:,1), '.','DisplayName', '{\lambda}^* Model');
hold on
p2_1 = plot(t1, best_slip_RLS(:,1),'.', 'DisplayName', '{\lambda}^* RLS');
grid on
xlabel('Time (sec)')
ylabel('best slip {\lambda}')
legend([p1,p2_1])
title('Predizione Regressore RLS')
set(gca, 'ylim', [0, 1]);


figure(60)
subplot(2,1,1)
plot(t1,dev_std_Torch_Dropout(:,1), '.');
grid on
xlabel('Time (sec)')
ylabel('{\sigma} normalizzata')
title('Deviazione standard normalizzata')

subplot(2,1,2)
plot(t1,dev_std_Torch_Dropout(:,2), '.');
grid on
xlabel('Time (sec)')
ylabel('{\sigma} ')
title('Deviazione standard')





% prendo i  theta computati da RLS che danno il residuo minore tra lambda* e lambda
% e li uso per generare la  funzione de coeeff di attrito mu (stimato)attraverso i lambda generati
% vettore workspace'in'.

 [~, idx_theta_best] = min((best_slip_RLS(:,1)-best_slip_GT(:,1)));
 theta_final = RLS_param_estimates(idx_theta_best,:);

for j = 1:length(lambda_hat)
      H = [1 lambda_hat(j) exp(G1*lambda_hat(j)) exp(G2*lambda_hat(j)) exp(G3*lambda_hat(j))];   
     out(j) = H*theta_final';
end 

% genero un vettore di uni della stessa lunghezza dei theta cui
% moltiplicher i valori finali "ottimi" stimati per calcolare
% successivamente l'evoluizione dell'errore dei theta nel tempo.

thetas_final_vect = ones(length(RLS_param_estimates),length(RLS_param_estimates(1,:)));
for i = 1:length(thetas_final_vect)
    thetas_final_vect(i,:)= thetas_final_vect(i,:).*theta_final;
end

figure(70)
subplot(211)
suptitle('Theta RLS per regressore')
plot(t1,RLS_param_estimates)
xlabel('Time (sec)')
ylabel('Values')
title('Theta stimati')
grid on
legend('{\theta}1','{\theta}2','{\theta}3', '{\theta}4','{\theta}5')
subplot(212)
plot(t1,(thetas_final_vect-RLS_param_estimates))
grid on
xlabel('Time (sec)')
ylabel('Residuals')
string_title = ['Errore Thetas stimati rispetto migliori fissati: ',  mat2str(theta_final)] ;
title(string_title)
legend('{\theta}1','{\theta}2','{\theta}3', '{\theta}4','{\theta}5')





