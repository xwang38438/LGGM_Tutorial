# python3 dataset_test.py dataset.name="asn" dataset.sample="seed" general.name="asn-seed uniform" model.transition="uniform" general.gpus=[0] train.batch_size=12 train.accumulate_grad_batches=4 general.setting='train_scratch'

# python3 dataset_test.py dataset.name="reddit" general.name="reddit_maxN150" model.transition="uniform" general.gpus=[0] train.batch_size=12 train.accumulate_grad_batches=4 general.setting='train_scratch' train.n_epochs=300



python3 main.py dataset.name="reddit_onehot" general.name="reddit_onehot" model.transition="uniform" general.gpus=[0] train.batch_size=16 train.accumulate_grad_batches=4 general.setting='train_scratch' train.n_epochs=300

