# NewYorkScapes Brown Brothers NYPL Collection Handwritten Text OCR CNN/RNN Models
This branch was created by [Ying Wang](https://github.com/YingWANGG) (yw3076@nyu.edu), built on the previous work of [Hong Gong](https://github.com/TRokieG) and [Lizhong Wang](https://github.com/Lizhong1024). Porject website is available at [https://brownbros.newyorkscapes.org/](https://brownbros.newyorkscapes.org).

## How to run the code on NYU HPC with GPU

1. Follow this [Greene Tutorial](https://github.com/nyu-dl/cluster-support/tree/master/greene) to setup singularity. Then, `pip install -r requirements.txt` to install all required libraries for this project (you need the rw flag here for installation).
2. Transfer the Brown Brothers data to Greene. You probably need to use `scp` or other data transferring tools. For more information, check [HPC Official Documentation - Data Transfers](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/data-transfers?authuser=0). Due to the constraint on file size and number on Greene, a better practice is to build a single [squashFS file](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/squash-file-system-and-singularity?authuser=0) that can be shared among researchers. 
3. Modify the file path and email address in `crnn/bb-train.s` accordingly. Note that --nv flag is needed for GPU. Usually, the training with 1 GPU will take around 30 minutes, which siginificantly accelerates the training.





