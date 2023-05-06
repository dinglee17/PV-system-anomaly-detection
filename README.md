# SCVAE

This is the codebase for [SCVAE]().

# Usage

This section of the README walks through how to train and evaluate the SCVAE model.

## Preparing Data

The training code reads time series from a directory of .csv files. In the [data](data) folder, we have provided synthesized data including normal data and 6 types of anoamly data.

For creating your own data, simply prepare the time series data in the same format. The .csv files should at least contain colums of "instantaneous_output_power","instantaneous_global_radiation","clearsky_Tamb" for power yeild, irradiation intensity and environment temperature data. You should also scale the data at the time dimension using Z-Score standardization.

## Training

To train your model, you should first decide some hyperparameters.Here are some reasonable defaults for a baseline:

```
"--h_dim 512 --z_dim 128 --mode 2 --learning_rate 1e-5 --batch_size 256 --seq_shape (-1, 12, 6)"
```

Once you have setup your hyper-parameters, you can run an experiment like so:

```
python scripts/SCVAE_train.py --h_dim 512 --z_dim 128
```

## Evaluating

The above training script saves checkpoints to `.pth` files in the "saved_model" directory.

Once you have a path to your model, you can evaluate your model's reconstruction results using synthesized data like so:

```
python scripts/SCVAE_eval.py --model_path /path/to/model.pth --anomaly_type all
```

## Getting latent variables & Clustering and classfication

Once you have a path to your model, you can also get the latent variable s of synthesized data as its representations like so. The latent variables are saved as a set of .npy files in the "re_z" folder.

```
python scripts/get_latent_var.py --model_path /path/to/model.pth
```

Once you have a set of latent variables files, you can do some clustering and classification experiments to analysis these latent variables like so:

```
python scripts/cluster_and_classify.py --latent_var_path ../rez --task clustering --anomaly_type whole_anomaly
python scripts/cluster_and_classify.py latent_var_path ../rez --task classification --anomaly_type whole_anomaly --classifier XGB
```
