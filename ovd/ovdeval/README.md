---
license: openrail
---

<h1 align="center"> OVDEval </h1>
<h2 align="center"> A Comprehensive Evaluation Benchmark for Open-Vocabulary Detection</h2>
<p align="center">
 <a href="https://arxiv.org/abs/2308.13177"><strong> [Paper 📄] </strong></a>
</p>

## Dataset Description

**OVDEval** is a new benchmark for OVD model, which includes 9 sub-tasks and introduces evaluations on commonsense knowledge, attribute understanding, position understanding, object relation comprehension, and more. The dataset is meticulously created to provide hard negatives that challenge models' true understanding of visual and linguistic input. Additionally, we identify a problem with the popular Average Precision (AP) metric when benchmarking models on these fine-grained label datasets and propose a new metric called **Non-Maximum Suppression Average Precision (NMS-AP)** to address this issue.


## Data Details

![image/png](https://cdn-uploads.huggingface.co/production/uploads/658a2e94991d8e7fb24f7688/ngOkek9wJdppyxPB0xZ8Q.png)


## Dataset Structure

```python
{
  "categories": [
    {
      "supercategory": "object",
      "id": 0,
      "name": "computer without screen on"
    },
    {
      "supercategory": "object",
      "id": 1,
      "name": "computer with screen on"
    }
]
  "annotations": [
    {
      "id": 0,
      "bbox": [
        111,
        117,
        99,
        75
      ],
      "category_id": 0,
      "image_id": 0,
      "iscrowd": 0,
      "area": 7523
    }]
  "images": [
    {
      "file_name": "64d22c6fe4b011b0db94b993.jpg",
      "id": 0,
      "height": 254,
      "width": 340,
      "text": [
        "computer without screen on"  # "text" represents the annotated positive labels of this image.
      ],
      "neg_text": [
        "computer with screen on" # "neg_text" contains fine-grained hard negative labels which are generated according specific sub-tasks.
      ]
    }]
}

```

## How to use it

Reference https://github.com/om-ai-lab/OVDEval

## Languages

The dataset contains questions in English and code solutions in Python.

## Citation Information
If you find our data, or code helpful, please cite the original paper:

```
@article{yao2023evaluate,
  title={How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection},
  author={Yao, Yiyang and Liu, Peng and Zhao, Tiancheng and Zhang, Qianqian and Liao, Jiajia and Fang, Chunxin and Lee, Kyusong and Wang, Qing},
  journal={arXiv preprint arXiv:2308.13177},
  year={2023}
}
```