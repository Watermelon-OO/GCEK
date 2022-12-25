## GCEK
[Exploiting global context and external knowledge for distantly supervised relation extraction](https://www.sciencedirect.com/science/article/pii/S0950705122012916)

## Requirements
- python >= 3.7
- mxnet	>= 1.5.1	

## Run Code 
1. preprocess NYT data 

    NYT 570K
    ```shell
    cd codes/data/NYT_data/NYT_522611/extract_cpp/
    unzip -d ./ NYT_570088.zip
    ./extract
    ```
    NYT 520K
    ```shell
    cd codes/data/NYT_data/NYT_570088/extract_cpp/
    unzip -d ./ NYT_522611.zip
    ./extract
    ```
2. generate train data and test data
   
   ```shell
   cd dataProcess
   python NYTData.py
   ```
3. train model 
   
   ```shell
   cd codes
   ```
   
   train with cpu
   ```shell
   python main.py -n 50 --lib GCEK -m GCEK -p GCEK_520K -o momentum --data_version 520
   ```
   
   train with gpu
   ```shell
   python main.py -c 0 -n 50 --lib GCEK -m GCEK -p GCEK_520K -o momentum --data_version 520
   ```
   
## Cite
Please cite as:
```
@article{GCEK,
title = {Exploiting global context and external knowledge for distantly supervised relation extraction},
journal = {Knowledge-Based Systems},
volume = {261},
pages = {110195},
year = {2023},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2022.110195},
url = {https://www.sciencedirect.com/science/article/pii/S0950705122012916},
author = {Jianwei Gao and Huaiyu Wan and Youfang Lin},
}
```