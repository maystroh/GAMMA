#!/usr/bin/env bash


# # _classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_22-13-33/epoch_174_batch_13.pth.tar
# python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=tf_efficientnetv2_b3 --task=classification --n_classes=1 --input_fundus_W=900 --input_fundus_H=900 --batch_size=6 --modalities_to_load Fundus --group_classes --augment_data &> tf_efficientnetv2_b3_binary_fundus_augemnt.log
# # _classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_23-10-17/epoch_425_batch_13.pth.tar
# python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=tf_efficientnetv2_b3 --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=6 --modalities_to_load Fundus --augment_data &> tf_efficientnetv2_b3_multiclass_fundus_augemnt.log


# ./weights/_classification_Fundus_tf_efficientnetv2_b3_512_512_Jul29_20-44-26/epoch_401_batch_7.pth.tar
#python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=tf_efficientnetv2_b3 --task=classification --n_classes=1 --input_fundus_W=512 --input_fundus_H=512 --batch_size=10 --modalities_to_load Fundus --group_classes --patch_evaluation --augment_data &> tf_efficientnetv2_b3_binary_fundus_augemnt_patch.log

#IT DID NOT WORK...
#python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=tf_efficientnetv2_b3 --task=classification --n_classes=3 --input_fundus_W=512 --input_fundus_H=512 --batch_size=10 --modalities_to_load Fundus --patch_evaluation --augment_data &> tf_efficientnetv2_b3_multiclass_fundus_augemnt_patch.log


# ./weights/_classification_Fundus_repvgg_b3_900_900_Jul29_23-10-16/epoch_200_batch_13.pth.tar
#python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=repvgg_b3 --task=classification --n_classes=1 --input_fundus_W=900 --input_fundus_H=900 --batch_size=6 --modalities_to_load Fundus --augment_data --group_classes  &> RepVGG-B3_binary_fundus_augemnt.log
# ./weights/_classification_Fundus_repvgg_b3_900_900_Jul30_09-08-30/epoch_127_batch_13.pth.tar
#python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=repvgg_b3 --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=6 --modalities_to_load Fundus --augment_data &> RepVGG-B3_multiclass_fundus_augemnt.log


#tf_efficientnetv2_m_in21ft1k
# ./weights/_classification_Fundus_tf_efficientnetv2_m_in21ft1k_900_900_Jul30_08-01-03/......
#srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=tf_efficientnetv2_m_in21ft1k --task=classification --n_classes=1 --input_fundus_W=900 --input_fundus_H=900 --batch_size=3 --modalities_to_load Fundus --augment_data --group_classes 
#IT DID NOT WORK...
#srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=tf_efficientnetv2_m_in21ft1k --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=6 --modalities_to_load Fundus --augment_data
#IT DID NOT WORK...
#srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=tf_efficientnetv2_m_in21ft1k --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=6 --modalities_to_load Fundus --augment_data --lr=0.001


# tf_efficientnet_b6
#srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=tf_efficientnet_b6 --task=classification --n_classes=1 --input_fundus_W=900 --input_fundus_H=900 --batch_size=3 --modalities_to_load Fundus --augment_data --group_classes 
#IT DID NOT WORK...
#srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=tf_efficientnet_b6 --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=3 --modalities_to_load Fundus --augment_data 

# dm_nfnet_f4
#srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=dm_nfnet_f5 --task=classification --n_classes=1 --input_fundus_W=900 --input_fundus_H=900 --batch_size=3 --modalities_to_load Fundus --augment_data --group_classes 
#srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=dm_nfnet_f4 --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=3 --modalities_to_load Fundus --augment_data 


# vit_large_patch16_384
#srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=vit_large_patch16_384 --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=3 --modalities_to_load Fundus --augment_data 


# gluon_resnext101_64x4d
# srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --model_2D=gluon_resnext101_64x4d --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=6 --modalities_to_load Fundus --augment_data 


########################### OCT Classification ####################
# srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py  --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=3 --model_3D=MedNet50 --modalities_to_load OCT --augment_data --patch_evaluation

# srun singularity exec --nv /home/hassan/Containers/torch_latest.simg python trainer.py  --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=64 --input_struct_H=224 --input_struct_W=512 --task=classification --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --batch_size=3 --model_3D=MedNet50 --modalities_to_load OCT --augment_data --patch_evaluation