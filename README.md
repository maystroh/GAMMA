# GAMMA
It is a solution to [GAMMA](https://aistudio.baidu.com/aistudio/competition/detail/90) (a challenge on applying deep learning for medical multi-modality data).

Please refer to the link above to get the data of the challenge. The dataset is a multi modality data: OCT for 200 patients along with their fundus image. 

This solution is based on 3D and 2D deep learning classification methods. 

##### Prerequisites:
- `Pytorch`
- `timm`

Please use `environment.yml` to have all the prerequisites

##### Data preprocessing
- `python gamma-prepare-data.py --input_dirs=../GAMMA_data/validation_data --depth_oct=256 --height_oct=224  --width_oct=512 --height_fundus=900 --width_fundus=900 --gamma_task=classification`
There is an option to segment the retina layers in the OCT volumes `--segment_retina_layers`. We can not provide the models trained for this part

##### Distributed training on 2 nodes, 4 GPUs each:
- `singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer_multidist.py --nodes=2 --gpus=4 --nr=0 --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --model_3D=MedNet50 --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --task=classification --n_classes=1 --input_fundus_W=900 --input_fundus_H=900 --batch_size=1 --modalities_to_load OCT --group_classes --n_epochs=5 --dry_run`

##### Solo training:
- `singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256  --task=classification --batch_size=1 --n_classes=3  --input_fundus_W=512  --input_fundus_H=512 --modalities_to_load OCT --augment_data  --group_classes --patch_evaluation`
- `singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256  --task=classification --batch_size=1 --n_classes=3  --model_2D=vit_large_patch16_384 --input_fundus_W=900  --input_fundus_H=900 --modalities_to_load Fundus --augment_data  --group_classes`

##### Refuge2020 results:
- This folder contains the presentations and the solutions for the top ranked teams in the final and semi-final phases 

# Copyright
Copyright © 2018-2019 [LaTIM U1101 Inserm](http://latim.univ-brest.fr/)

