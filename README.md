# S2D: Enhancing Zero-shot Cross-lingual Event Argument Extraction with Semantic Knowledge

## Setup

- Python=3.8.18

```bash
$ conda env create -f environment.yml
```
## Data and Preprocessing

- EAE data:https://github.com/PlusLabNLP/X-Gear

- SRL data:https://github.com/zzsfornlp/zmsp/
  
- How to convert SRL data to EAE data：

  First when you have completed the above two steps of data preprocessing, you need to manually write code to convert the SRL data format to the format of the processed EAE dataset.

- How to splice the SRL data from the previous step into the EAE dataset:

  After you have processed the first step, you will get the folder “finetuned_data”, in which you need to add the processed SRL dataset in the same format as the processed EAE dataset to the parts of “train_all.pkl” and “vocab.json”  .


## Details：

- The file “S2D” is our main model code.
- The file “config-large” is the parameter file.

## Run:
  You can change the parameters in the file "config-large" and run "train_ace_xx_xx.py" (xx is replaced by language) in "S2D" to do the training.
  
