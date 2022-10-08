function [best_slip, row_dataset]= fcn(slip_buffer, mu_buffer)
%#codegen
coder.extrinsic('py.wrapperML.predict')
coder.extrinsic('py.numpy.array')
slip = slip_buffer
mu = mu_buffer
row_dataset = reshape([slip mu]',[],1)

npA = py.numpy.array(row_dataset(:).');

best_slip = 0.0;
best_slip = py.wrapperML.predict([npA]);

