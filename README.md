[![Python-Versions](https://img.shields.io/badge/python-3.10-blue.svg)]()
[![Open in HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Open_in_HuggingFace-orange)](https://huggingface.co/IMISLab/)
[![Software-License](https://img.shields.io/badge/License-Apache--2.0-green)](https://github.com/NC0DER/GreekReddit/blob/main/LICENSE)

# GreekReddit
<img src="Greek Reddit icon.svg" width="200"/>  

This repository hosts code for the article:
* [Mastrokostas, C., Giarelis, N., & Karacapilidis, N. (2024). Social Media Topic Classification on Greek Reddit](https://www.mdpi.com/2078-2489/15/9/521)


## About
This repository stores the data crawling and processing code for the `GreekReddit` dataset, as well as the training and evaluation code for the proposed models.
The proposed models were trained and evaluated on `GreekReddit`.
The dataset and best-performing models are hosted on [HuggingFace](https://huggingface.co/IMISLab).


## Installation
```
pip install requirements.txt
```

## Example Code
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_name = 'IMISLab/Greek-Reddit-BERT'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name) 

topic_classifier = pipeline(
    'text-classification',
    device = 'cpu',
    model = model,
    tokenizer = tokenizer,
    truncation = True,
    max_length = 512
)
    
text = 'Άλλες οικονομίες, όπως η Κίνα, προσπαθούν να διατηρούν την αξία του νομίσματος τους χαμηλά ώστε να καταστήσουν τις εξαγωγές τους πιο ελκυστικές στο εξωτερικό. Γιατί όμως θεωρούμε πως η πτωτική πορεία της Τουρκικής λίρας είναι η ""αχίλλειος πτέρνα"" της Τουρκίας;'
output = topic_classifier(text)
print(output[0]['label'])
```

## Citation
```
@article{mastrokostas2024social,
  title={Social Media Topic Classification on Greek Reddit},
  author={Mastrokostas, Charalampos and Giarelis, Nikolaos and Karacapilidis, Nikos},
  journal={Information},
  volume={15},
  number={9},
  pages={521},
  year={2024},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

## Contributors
* Nikolaos Giarelis (giarelis@ceid.upatras.gr)
* Charalampos Mastrokostas (cmastrokostas@ac.upatras.gr)
* Nikos Karacapilidis (karacap@upatras.gr)
