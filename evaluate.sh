#!/usr/bin/env bash

# _classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_20-22-43/epoch_23_batch_13.pth.tar
python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=1 --input_fundus_W=900 --input_fundus_H=900 --model_2D=tf_efficientnetv2_b3 --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_20-22-43/epoch_23_batch_13.pth.tar --modalities_to_load Fundus --group_classes 

# _classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_21-18-10/epoch_39_batch_13.pth.tar
python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --model_2D=tf_efficientnetv2_b3 --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_21-18-10/epoch_39_batch_13.pth.tar  --modalities_to_load Fundus 

# _classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_22-13-33/epoch_174_batch_13.pth.tar
python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=1 --input_fundus_W=900 --input_fundus_H=900 --model_2D=tf_efficientnetv2_b3 --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_22-13-33/epoch_174_batch_13.pth.tar --modalities_to_load Fundus --group_classes --augment_data 

# _classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_23-10-17/epoch_425_batch_13.pth.tar
python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --model_2D=tf_efficientnetv2_b3 --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_23-10-17/epoch_425_batch_13.pth.tar  --modalities_to_load Fundus --augment_data 

# # throw an exception
# python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=1 --input_fundus_W=512 --input_fundus_H=512 --model_2D=tf_efficientnetv2_b3 --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_22-13-33/epoch_174_batch_13.pth.tar 
# --modalities_to_load Fundus --group_classes --patch_evaluation --augment_data 
# # Dev kappa is zero
# python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=3 --input_fundus_W=512 --input_fundus_H=512 --model_2D=tf_efficientnetv2_b3 --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_22-13-33/epoch_174_batch_13.pth.tar  
# --modalities_to_load Fundus --patch_evaluation --augment_data 


# python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=1 --input_fundus_W=900 --input_fundus_H=900 --model_2D=tf_efficientnetv2_l_in21ft1k --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_22-13-33/epoch_174_batch_13.pth.tar
#  --modalities_to_load Fundus --group_classes --augment_data 
# python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=3 --input_fundus_W=900 --input_fundus_H=900 --model_2D=tf_efficientnetv2_l_in21ft1k --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_22-13-33/epoch_174_batch_13.pth.tar
#  --modalities_to_load Fundus --augment_data 

# python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=1 --input_fundus_W=512 --input_fundus_H=512 ---model_2D=tf_efficientnetv2_l_in21ft1k -model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_22-13-33/epoch_174_batch_13.pth.tar
#  --modalities_to_load Fundus --group_classes --patch_evaluation --augment_data 
# python evaluate.py --data_root=../GAMMA_data/training_data/classification_F_900_900_OCT_512_224_256_Seg_True --input_struct_D=256 --input_struct_H=224 --input_struct_W=256 --n_classes=3 --input_fundus_W=512 --input_fundus_H=512 --model_2D=tf_efficientnetv2_l_in21ft1k --model_path=./weights/_classification_Fundus_tf_efficientnetv2_b3_900_900_Jul28_22-13-33/epoch_174_batch_13.pth.tar
#  --modalities_to_load Fundus --patch_evaluation --augment_data 

