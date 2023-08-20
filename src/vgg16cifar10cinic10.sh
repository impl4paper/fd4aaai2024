echo "The name of script: $0, gpu: $1, noidd: $2"
python pri_cen_label_fedmd.py --name vgg16 --method local  --gpu $1 --noidd $2
python pri_cen_label_fedmd.py --name vgg16 --method avglabels  --gpu $1 --noidd $2
python pri_cen_label_fedmd.py --name vgg16 --method avglogits  --gpu $1 --noidd $2
python pri_cen_label_fedmd.py --name vgg16 --method avglogits-dp  --gpu $1 --noidd $2

