# python3 dataset_test.py dataset.name="asn" dataset.sample="seed" general.name="asn-seed uniform" model.transition="uniform" general.gpus=[0] train.batch_size=12 train.accumulate_grad_batches=4 general.setting='train_scratch'

python3 main_gad.py dataset.name="reddit_continuous" general.name="reddit_onehot_conguide" model.transition="uniform" general.gpus=[0] train.batch_size=16 train.accumulate_grad_batches=4 general.setting='train_scratch' train.n_epochs=300



