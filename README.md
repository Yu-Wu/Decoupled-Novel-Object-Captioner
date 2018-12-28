# [Decoupled Novel Object Captioner](https://arxiv.org/pdf/1804.03803.pdf)

TensorFlow code for our paper [[Link]](https://arxiv.org/pdf/1804.03803.pdf).

## Preparation
### Dependencies
- Python 3.6
- Tensorflow (version >= 1.4.1)
- nltk, tqdm

### Prepare data
Please follow the instructions in [prepare_data](https://github.com/Yu-Wu/Decoupled-Novel-Object-Captioner/tree/master/prepare_data/).

## Train
```shell
python run.py --stage train
```
It takes about one hour to train the model with a Nvidia V100 GPU. 

## Test
```shell
python run.py --stage test
```

## Reference

Please cite the following paper in your publications if it helps your research:

    @inproceedings{wu2018decoupled,
      title     = {Decoupled Novel Object Captioner},
      author    = {Wu, Yu and Zhu, Linchao and Jiang, Lu and Yang, Yi},
      booktitle = {Proceedings of the 2018 ACM on Multimedia Conference (ACM MM)},
      year      = {2018}
    }


## Contact

To report issues for this code, please open an issue on the [issues tracker](https://github.com/Yu-Wu/Decoupled-Novel-Object-Captioner/issues).

If you have further questions about this paper, please do not hesitate to contact me. 

[Yu Wu's Homepage](https://yu-wu.net)
