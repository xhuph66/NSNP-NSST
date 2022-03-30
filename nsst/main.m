dataset_name = 'nasdaq100';
file=['./Normalize_dataset/',dataset_name,'.csv'];
dataset = csvread(file)
pfilt = 'maxflat';
shear_parameters.dcomp =[0,1,3]; %[0,1,3] [3,3,4,4]   %[4,4,3,3]
shear_parameters.dsize =[1,2,8]; %[8,8,16,16] %[32,32,16,16]
[d ,shear_f]=nsst_dec2(dataset,shear_parameters,pfilt);

%%
csvwrite(['./dataset_coeffs/',dataset_name,'/one/01_',dataset_name,'.csv'],d{1,1})
csvwrite(['./dataset_coeffs/',dataset_name,'/one/02_',dataset_name,'.csv'],d{1,2})
csvwrite(['./dataset_coeffs/',dataset_name,'/two/1_',dataset_name,'.csv'],d{1,3}(:,:,1))
csvwrite(['./dataset_coeffs/',dataset_name,'/two/2_',dataset_name,'.csv'],d{1,3}(:,:,2))
for k=1:2^3
    name=['./dataset_coeffs/',dataset_name,'/three/',num2str(k),'_',dataset_name,'.csv']
    csvwrite(name,d{1,4}(:,:,k))
end
%%
% read the data after NSNP system transform 
d{1,1}=csvread(['./re_coffis/',dataset_name,'/one/01_',dataset_name,'.csv'])
d{1,2}(:,:)=csvread(['./re_coffis/',dataset_name,'/one/02_',dataset_name,'.csv'])
d{1,3}(:,:,1)=csvread(['./re_coffis/',dataset_name,'/two/01_',dataset_name,'.csv'])
d{1,3}(:,:,2)=csvread(['./re_coffis/',dataset_name,'/two/02_',dataset_name,'.csv'])
for k=1:2^3
    name=['./re_coffis/',dataset_name,'/three/',num2str(k),'_',dataset_name,'.csv']
    d{1,4}(:,:,k)=csvread(name)
end
%%
new_predicted = nsst_rec2(d,shear_f,pfilt)
csvwrite(['./reconstruct/',dataset_name,'/',dataset_name,'.csv'],new_predicted)