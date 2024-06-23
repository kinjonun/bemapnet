
<summary><b> Training and Evluation</b></summary>


- **a. Model Training**
  ```shell
  bash run.sh train bemapnet_nuscenes_swint 30  # default: 8GPUs, bs=1, epochs=30
  bash run.sh train bemapnet_nuscenes_res50 2 0 1 
  bash run.sh train bemapnet_av2_res50 2 0 1
  ```

- **b. Model Evaluation**
  ```shell
  bash run.sh test bemapnet_nuscenes_swint ${checkpoint-path}
  bash run.sh test bemapnet_nuscenes_res50 outputs/bemapnet_nuscenes_res50/2024-05-03T10\:33\:35/dump_model/checkpoint_epoch_29.pth
  bash run.sh test bemapnet_av2_res50 
  ```

- **c. Reproduce with one command**
  ```shell
  bash run.sh reproduce bemapnet_nuscenes_swint
  ```
