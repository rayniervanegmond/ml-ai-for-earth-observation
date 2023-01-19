# Cancer Tissue Detection using CNN


#### Installing Required libraries

### Step 1.1- Please refer to the "Technical Requirements" section in the book for the neccessary packages to be installed. Please note that chapter has differnt Pytorch Lightning version and thus diff torch dependancies. Some functions may not work with other versions than what is tested below, so please ensure correct versions.



```python
#Step1 Please refer to the "Technical Requirements" section in the book for the neccessary packages to be installed here
# !pip install torch==1.10.0 torchvision==0.11.1 torchtext==0.11.0 torchaudio==0.10.0 --quiet
# !pip install pytorch-lightning==1.5.2 --quiet
```


```python
!pip install opendatasets --upgrade --quiet
```

## **Step 1.2 Import the neccessary packages. Please refer to the section, Importing the packages in the book for further details. **


```python
#refer to book for correct version of package and import here
import os
import shutil
import opendatasets as od
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
```


```python
print("pandas version:",pd.__version__)
print("numpy version:",np.__version__)
#print("seaborn version:",sns.__version__)
print("torch version:",torch.__version__)
print("pytorch ligthening version:",pl.__version__)


```

    pandas version: 1.5.2
    numpy version: 1.24.1
    torch version: 1.13.1
    pytorch ligthening version: 1.8.6


# **Step 2.1 - Load the dataset- Please refer to the book in section "collecting the dataset" for the proper way to collect the dataset and steps to import it. **

Refer to section "Collecting the dataset and copy code here "
Be careful with the dash at the end of the line when copy pasting the dataset_url

We will be using the kaggle cli application with the command


```python
!kaggle competitions download -c ../data/histopathologic-cancer-detection
```

which needs to be executed in the directory where the data should be ocated for the notebook.


```python
# dataset_url = 'https://www.kaggle.com/c/histopathologic-cancer-detection'
# od.download(dataset_url)
```

There is a separate csv file which contains only the labels. Let's read the dataset and see the head of the dataframe in pandas


```python
cancer_labels = pd.read_csv('../data/histopathologic-cancer-detection/train_labels.csv')
cancer_labels.head()
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Cell In[7], line 1
    ----> 1 cancer_labels = pd.read_csv('data/histopathologic-cancer-detection/train_labels.csv')
          2 cancer_labels.head()


    File ~/miniconda3/envs/pytorch13/lib/python3.10/site-packages/pandas/util/_decorators.py:211, in deprecate_kwarg.<locals>._deprecate_kwarg.<locals>.wrapper(*args, **kwargs)
        209     else:
        210         kwargs[new_arg_name] = new_arg_value
    --> 211 return func(*args, **kwargs)


    File ~/miniconda3/envs/pytorch13/lib/python3.10/site-packages/pandas/util/_decorators.py:331, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        325 if len(args) > num_allow_args:
        326     warnings.warn(
        327         msg.format(arguments=_format_argument_list(allow_args)),
        328         FutureWarning,
        329         stacklevel=find_stack_level(),
        330     )
    --> 331 return func(*args, **kwargs)


    File ~/miniconda3/envs/pytorch13/lib/python3.10/site-packages/pandas/io/parsers/readers.py:950, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        935 kwds_defaults = _refine_defaults_read(
        936     dialect,
        937     delimiter,
       (...)
        946     defaults={"delimiter": ","},
        947 )
        948 kwds.update(kwds_defaults)
    --> 950 return _read(filepath_or_buffer, kwds)


    File ~/miniconda3/envs/pytorch13/lib/python3.10/site-packages/pandas/io/parsers/readers.py:605, in _read(filepath_or_buffer, kwds)
        602 _validate_names(kwds.get("names", None))
        604 # Create the parser.
    --> 605 parser = TextFileReader(filepath_or_buffer, **kwds)
        607 if chunksize or iterator:
        608     return parser


    File ~/miniconda3/envs/pytorch13/lib/python3.10/site-packages/pandas/io/parsers/readers.py:1442, in TextFileReader.__init__(self, f, engine, **kwds)
       1439     self.options["has_index_names"] = kwds["has_index_names"]
       1441 self.handles: IOHandles | None = None
    -> 1442 self._engine = self._make_engine(f, self.engine)


    File ~/miniconda3/envs/pytorch13/lib/python3.10/site-packages/pandas/io/parsers/readers.py:1735, in TextFileReader._make_engine(self, f, engine)
       1733     if "b" not in mode:
       1734         mode += "b"
    -> 1735 self.handles = get_handle(
       1736     f,
       1737     mode,
       1738     encoding=self.options.get("encoding", None),
       1739     compression=self.options.get("compression", None),
       1740     memory_map=self.options.get("memory_map", False),
       1741     is_text=is_text,
       1742     errors=self.options.get("encoding_errors", "strict"),
       1743     storage_options=self.options.get("storage_options", None),
       1744 )
       1745 assert self.handles is not None
       1746 f = self.handles.handle


    File ~/miniconda3/envs/pytorch13/lib/python3.10/site-packages/pandas/io/common.py:856, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        851 elif isinstance(handle, str):
        852     # Check whether the filename is to be opened in binary mode.
        853     # Binary mode does not support 'encoding' and 'newline'.
        854     if ioargs.encoding and "b" not in ioargs.mode:
        855         # Encoding
    --> 856         handle = open(
        857             handle,
        858             ioargs.mode,
        859             encoding=ioargs.encoding,
        860             errors=errors,
        861             newline="",
        862         )
        863     else:
        864         # Binary mode
        865         handle = open(handle, ioargs.mode)


    FileNotFoundError: [Errno 2] No such file or directory: 'data/histopathologic-cancer-detection/train_labels.csv'



```python
cancer_labels['label'].value_counts()
```

There are 130908 normal cases (0) and and 89117 abnormal (cancerous) cases (1) 

*   List item
*   List item

which is not highly unbalanced. 


```python
print('No. of images in training dataset: ', len(os.listdir("histopathologic-cancer-detection/train")))
print('No. of images in testing dataset: ', len(os.listdir("histopathologic-cancer-detection/test")))
```

This is a huge dataset which requires a lot of compute time and resources so for the purpose of learning our first basic image classification model, we will downsample it to 5000 images and then split it into training and testing dataset.


```python
# Setting seed to make the results replicable
np.random.seed(0)
train_imgs_orig = os.listdir("histopathologic-cancer-detection/train")
selected_image_list = []
for img in np.random.choice(train_imgs_orig, 10000):
  selected_image_list.append(img)
len(selected_image_list)
```


```python
selected_image_list[0]
```


```python
fig = plt.figure(figsize=(25, 6))
for idx, img in enumerate(np.random.choice(selected_image_list, 20)):
    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    im = Image.open("histopathologic-cancer-detection/train/" + img)
    plt.imshow(im)
    lab = cancer_labels.loc[cancer_labels['id'] == img.split('.')[0], 'label'].values[0]
    ax.set_title(f'Label: {lab}')
```


```python
np.random.seed(0)
np.random.shuffle(selected_image_list)
cancer_train_idx = selected_image_list[:8000]
cancer_test_idx = selected_image_list[8000:]
print("Number of images in the downsampled training dataset: ", len(cancer_train_idx))
print("Number of images in the downsampled testing dataset: ", len(cancer_test_idx))
```

### Processing the dataset
The following information has been provided on the Kaggle and the Github where the dataset is hosted - "A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable the design of fully-convolutional models that do not use any zero-padding, to ensure consistent behavior when applied to a whole-slide image."


```python
from google.colab import drive
drive.mount('/content/gdrive')
```


```python
 #cd "/content/gdrive/MyDrive/Colab Notebooks"
```


```python
#os.mkdir('/content/histopathologic-cancer-detection/train_dataset/')
```


```python
os.mkdir('/content/histopathologic-cancer-detection/train_dataset/')
for fname in cancer_train_idx:
  src = os.path.join('histopathologic-cancer-detection/train', fname)
  dst = os.path.join('/content/histopathologic-cancer-detection/train_dataset/', fname)
  shutil.copyfile(src, dst)
print('No. of images in downsampled training dataset: ', len(os.listdir("/content/histopathologic-cancer-detection/train_dataset/")))

```


```python
os.mkdir('/content/histopathologic-cancer-detection/test_dataset/')
for fname in cancer_test_idx:
  src = os.path.join('histopathologic-cancer-detection/train', fname)
  dst = os.path.join('/content/histopathologic-cancer-detection/test_dataset/', fname)
  shutil.copyfile(src, dst)
print('No. of images in downsampled testing dataset: ', len(os.listdir("/content/histopathologic-cancer-detection/test_dataset/")))
```


```python
data_T_train = T.Compose([
    T.CenterCrop(32),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    ])
data_T_test = T.Compose([
    T.CenterCrop(32),
    T.ToTensor(),
    ])
```


```python
# Extracting the labels for the images that were selected in the downsampled data
selected_image_labels = pd.DataFrame()
id_list = []
label_list = []

for img in selected_image_list:
  label_tuple = cancer_labels.loc[cancer_labels['id'] == img.split('.')[0]]
  id_list.append(label_tuple['id'].values[0])
  label_list.append(label_tuple['label'].values[0])
```


```python
selected_image_labels['id'] = id_list
selected_image_labels['label'] = label_list
selected_image_labels.head()
```


```python
# dictionary with labels and ids of train data
img_label_dict = {k:v for k, v in zip(selected_image_labels.id, selected_image_labels.label)}
```

Pytorch lightning expects data to be in folders with the classes. We cannot use the DataLoader module directly when all train images are in one folder without subfolders. So, we will write our custom function to carry out the loading. 



```python
class LoadCancerDataset(Dataset):
    def __init__(self, data_folder, 
                 transform = T.Compose([T.CenterCrop(32),T.ToTensor()]), dict_labels={}):
        self.data_folder = data_folder
        self.list_image_files = [s for s in os.listdir(data_folder)]
        self.transform = transform
        self.dict_labels = dict_labels
        self.labels = [dict_labels[i.split('.')[0]] for i in self.list_image_files]

    def __len__(self):
        return len(self.list_image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.list_image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)
        img_name_short = self.list_image_files[idx].split('.')[0]

        label = self.dict_labels[img_name_short]
        return image, label
```


```python
%%time
# Load train data 
train_set = LoadCancerDataset(data_folder='/content/histopathologic-cancer-detection/train_dataset/', 
                        # datatype='train', 
                        transform=data_T_train, dict_labels=img_label_dict)
```


```python
test_set = LoadCancerDataset(data_folder='/content/histopathologic-cancer-detection/test_dataset/', 
                         transform=data_T_test, dict_labels=img_label_dict)

```


```python
batch_size = 256

train_dataloader = DataLoader(train_set, batch_size, num_workers=2, pin_memory=True, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size, num_workers=2, pin_memory=True)
```


```python
class CNNImageClassifier(pl.LightningModule):

    def __init__(self, learning_rate = 0.001):
        super().__init__()

        self.learning_rate = learning_rate

        self.conv_layer1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv_layer2 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.fully_connected_1 =nn.Linear(in_features=16 * 16 * 6,out_features=1000)
        self.fully_connected_2 =nn.Linear(in_features=1000,out_features=500)
        self.fully_connected_3 =nn.Linear(in_features=500,out_features=250)
        self.fully_connected_4 =nn.Linear(in_features=250,out_features=120)
        self.fully_connected_5 =nn.Linear(in_features=120,out_features=60)
        self.fully_connected_6 =nn.Linear(in_features=60,out_features=2)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, input):
        output=self.conv_layer1(input)
        output=self.relu1(output)
        output=self.pool(output)
        output=self.conv_layer2(output)
        output=self.relu2(output)
        output=output.view(-1, 6*16*16)
        output = self.fully_connected_1(output)
        output = self.fully_connected_2(output)
        output = self.fully_connected_3(output)
        output = self.fully_connected_4(output)
        output = self.fully_connected_5(output)
        output = self.fully_connected_6(output)
        return output

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs) 
        train_accuracy = accuracy(outputs, targets)
        loss = self.loss(outputs, targets)
        self.log('train_accuracy', train_accuracy, prog_bar=True)
        self.log('train_loss', loss)
        return {"loss":loss, "train_accuracy":train_accuracy}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        test_accuracy = accuracy(outputs, targets)
        loss = self.loss(outputs, targets)
        self.log('test_accuracy', test_accuracy)
        return {"test_loss":loss, "test_accuracy":test_accuracy}

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = optim.Adam(params=params, lr = self.learning_rate)
        return optimizer

    # Calculate accuracy for each batch at a time
    def binary_accuracy(self, outputs, targets):
        _, outputs = torch.max(outputs,1)
        correct_results_sum = (outputs == targets).sum().float()
        acc = correct_results_sum/targets.shape[0]
        return acc

    def predict_step(self, batch, batch_idx ):
        return self(batch)
```


```python
model = CNNImageClassifier()

trainer = pl.Trainer(fast_dev_run=True, gpus=1)
trainer.fit(model, train_dataloaders=train_dataloader)
```


```python
ckpt_dir = "/content/gdrive/MyDrive/Colab Notebooks/cnn"
# ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_epochs=10)

model = CNNImageClassifier()
trainer = pl.Trainer(
    default_root_dir=ckpt_dir,
                     gpus=-1,
                    #  progress_bar_refresh_rate=30,
                        # callbacks=[ckpt_callback],
                        log_every_n_steps=25,
                        max_epochs=500)
trainer.fit(model, train_dataloaders=train_dataloader)
```


```python
trainer.test(test_dataloaders=test_dataloader)
```


```python
model.eval()
preds = []
for batch_i, (data, target) in enumerate(test_dataloader):
    data, target = data.cuda(), target.cuda()
    output = model.cuda()(data)

    pr = output[:,1].detach().cpu().numpy()
    for i in pr:
        preds.append(i)
```


```python
test_preds = pd.DataFrame({'imgs': test_set.list_image_files, 'labels':test_set.labels,  'preds': preds})

```


```python
test_preds['imgs'] = test_preds['imgs'].apply(lambda x: x.split('.')[0])

```


```python
test_preds.head()
```


```python
test_preds['predictions'] = 1
test_preds.loc[test_preds['preds'] < 0, 'predictions'] = 0
test_preds.shape
```


```python
test_preds.head()
```


```python
len(np.where(test_preds['labels'] == test_preds['predictions'])[0])/test_preds.shape[0]
```


```python

```
