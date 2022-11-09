## Traffic signs classificator using simple CNN

### Simple traffic signs classificator using convolutional neural networks

- Best accuracy achieved on testing set was ~96%
- Project was long forgotten and wandb logs lost
- Link to dataset : https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
- Main purpose was to get used to pytorch and other things used in deep learning

### Runnable code is stored in main.py file

- transform data flag specifies whether data should be resized and preprocessed, it reads data from ./data folder and saves it to ./resized_data
- the data is then read from such folder
- wandb logging is currently broken
- to run the code edit parameters needed and run:
```console
python3 main.py
```