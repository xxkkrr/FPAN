# Adapting User Preference to Online Feedback in Multi-round Conversational Recommendation

## Description

This repository is the source code of the paper *Adapting User Preference to Online Feedback in Multi-round Conversational Recommendation* implemented via PyTorch.

## Code
The codes of Yelp ([enumerated questions](https://github.com/xxkkrr/FPAN/tree/Yelp)) and LastFM ([binary questions](https://github.com/xxkkrr/FPAN/tree/LastFM)) are stored in different branch with minor differences.

## Requirement

- `torch==1.4.0`   
- `torch_geometric==1.4.3`
- `tqdm`
- `sklearn`

## Dataset
The dataset download link: `https://drive.google.com/file/d/1qUsdTGHPqawgJ04wx0YtfF8GCRRQYDav/view?usp=sharing`
The dataset we used is based on [Yelp](https://www.yelp.com/dataset/) and [LastFM](https://grouplens.org/datasets/hetrec-2011/) which are processed in [EAR WSDM'20](https://dl.acm.org/doi/10.1145/3336191.3371769)

After downloading Data.zip, unzip it and  put the files in Data/Yelp_data/ or  Data/LastFM_data/ to corresponding /data/ folder.

## Acknowledgement
Thanks the authors of [EAR WSDM'20](https://dl.acm.org/doi/10.1145/3336191.3371769)  for sharing their codes, datasets and model parameters.

Here is the [link](https://ear-conv-rec.github.io/) to their work.

