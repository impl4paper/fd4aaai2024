## Source codes for AAAI 2024.

### Some important directories or files are described as follows:
- dataprocess: Codes related to data processing, including various Non-IID settings.
- models: Codes related to  trained models, including CNN, VGGNet-16 and ResNet-18.
- third_party:  Codes related to LabelAvg.
- utils: Some useful tools.
- pri_cen_label_fedmd.py: Source codes of LabelAvg (contains source code for other baselines).
- L-Attack-Demo.ipynb: Source codes of  L-Attack.
- vgg16cifar10cinic10.sh: Demo of running scripts.

### Some important configurations are described as follows:
-- gpu: The id of GPU.
-- name: Define the name of the experiment.
-- num_users: The number of clients.
--local_ep: The number of local epochs.
-- distill_ep: The number of iterations during distillation.
-- pre_ep: The number of epochs during pre-training.
-- epochs: The number of epochs
-- local_bs: The size of mini-batch during local training.
-- global_bs: The size of mini-batch during distillation.
-- pre_bs: The size of mini-batch during pre-training.
-- noiid:  Used to configure the Non-IID seting.
-- method: Used to configure which algorithm to run. When this parameter is set to 'local', 'avglabels', 'avglogits' and 'avglogits-dp', the script executes the one-sided training, LabelAvg, FedMD, and FedMD-LDP procedures, respectively.
-- femnist: Conduct experiments on FEMNIST.
-- cifar100supcls: Conduct experiments on CIFAR-100.
-- cifar10: Conduct experiments on CIFAR-10.
