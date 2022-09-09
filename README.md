models for PV System anomaly detection

**SCVAE**

1. SCVAE_train.py for training model.   eg. python SCVAE_train.py

2. SCVAE_model1.py,SCVAE_model2.py,SCVAE_model3.py are three variants of SCVAE model; SCVAE_model2.py is used in trainning by default.

3. SCVAE_output.py for evaluating trained model.   eg. python SCVAE_output,py

4. saved_model folder contained the trained model files.

5. args can be tuned in SCVAE_train.py file, which are listed as followed:

    --dataset, type=str,default='data/site_30100005_series.npz', help='data file path'

    --reg, type=float, default=0,help='Smooth canonical intensity'  

    --batch_size, type=int,default=256, help="batch size"  

    --device, type=str, default='cuda',help="device, e.g. cpu, cuda"  

    --learning_rate, type=float,default=0.000001, help='Adam optimizer learning rate'  

    --print_every, type=int, default=1,help='The number of iterations in which the result is printed'  

    --n_epochs, type=int, default=10000,help='maximum number of iterations'  

    --h_dim, type=int, default=512,help='dimension in Neural network hidden layers '  

    --z_dim, type=int, default=128,help='dimensions of latent variable'  

    --test_ratio, type=float, default=1,help="the test ratio in data_set"  

    --normal_ratio, type=int,default=99.9, help="the nomal_ratio % in dataset"  

    --mode, type=int, default=2,help="the mode when train"  

    --is_predict, type=bool,default=False, help="whether predict ot not"  

    --is_prior, type=bool, default=False,help="whether prior reconstruct ot not"  

    --whether_get_z, type=bool,default=False, help="whether get hidden variable"  

    --is_scale_data, type=bool,default=False, help="whether get scale data"  

    --is_simulate_data, type=bool, default=True,help="whether use simulate data or predict data"  

    --anomaly_type_detected, type=str,default='all', help="The anomaly type to be detected"  
