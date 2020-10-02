# 2020Team20Project

## Dependencies

```
Keras 2.3.1
Tensorflow 2.1
PyTorch 1.4
```

## How to use

Please look into `main.py` to see which command-line arguments are available. For example, `python main.py -lm <PATH_TO_YOUR_MODEL> --auroc` loads the model specified in one of the directory, then uses the model to generate AUROC scores for all the labels using the validation set and saves AUROC graphs to the current directory.

Alternatively you can use `python models_keras.py` to train models in Keras. Parameters need to be adjusted in `model_keras.py`.
