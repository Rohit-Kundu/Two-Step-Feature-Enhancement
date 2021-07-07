[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cervical-cytology-classification-using-pca/image-classification-on-sipakmed)](https://paperswithcode.com/sota/image-classification-on-sipakmed?p=cervical-cytology-classification-using-pca)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cervical-cytology-classification-using-pca/image-classification-on-herlev)](https://paperswithcode.com/sota/image-classification-on-herlev?p=cervical-cytology-classification-using-pca)

# Cervical Cytology Classification Using PCA & GWO Enhanced Deep Features Selection

<img src="/overall.png" style="margin: 10px;">

Official Python Implementation of the paper titled ["Cervical Cytology Classification Using PCA & GWO Enhanced Deep Features Selection"](https://doi.org/10.1007/s42979-021-00741-2) published in the special issue "AI and Deep Learning Trends in Healthcare" of [SpringerNature Computer Science](https://www.springer.com/journal/42979).

## Requirements

To install the required dependencies run the following using the Command Prompt:

`pip install -r requirements.txt`

# Implementing the code for Cervical Cytology data

Similarly the script can be modified for extracting features from other models.

Structure the directory as follows:

```

.
+-- data
|   +-- .
|   +-- train
|   +-- val
+-- extract_features.py
+-- fitnessFUNs.py
+-- GWO.py
+-- main.py
+-- resnet50.csv
+-- selector.py
+-- solution.py
+-- transfer_functions_benchmark.py

```

To extract ResNet-50 features run the following script:

`python extract_features.py`

Run the following code for the feature set optimization:

`python main.py --num_csv 2`

Set `num_csv` to the number of features csv files you have. You will be asked to enter the names of the csv files upon executing the above code. Execute `python main.py -h` to get the details of all the available arguments.

## Citation

If this repository helps you in your research in any way, please cite our paper:

```
@article{Basak2021,
      author={Basak, Hritam and Kundu, Rohit and Chakraborty, Sukanta and Das, Nibaran},
      title={Cervical Cytology Classification Using PCA and GWO Enhanced Deep Features Selection},
      journal={SN Computer Science},
      year={2021},
      month={Jul},
      day={07},
      volume={2},
      number={5},
      pages={369},
      issn={2661-8907},
      doi={10.1007/s42979-021-00741-2},
      url={https://doi.org/10.1007/s42979-021-00741-2}
}
```
