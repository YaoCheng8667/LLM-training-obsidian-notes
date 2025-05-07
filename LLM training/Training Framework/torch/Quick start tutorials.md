# 📖 Contents[^1]

[[#1. Work with data]]
[[#2. Creating model]]
[[#3. Optimizing the Model Parameters]]
[[#4. Save models]]
[[#5. Loading models]]
## 1. Work with data

+ PyTorch has two [primitives to work with data](https://pytorch.org/docs/stable/data.html): `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`.
+ `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset`.
+ We pass the `Dataset` as an argument to `DataLoader`. This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading.

	- **🔖DataLoader**
		At the heart of PyTorch data loading utility is the [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") class. It represents a Python iterable over a dataset, with support for: 
		- [map-style and iterable-style datasets](https://pytorch.org/docs/stable/data.html#dataset-types),
		- [customizing data loading order](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler),
		- [automatic batching](https://pytorch.org/docs/stable/data.html#loading-batched-and-non-batched-data),
		- [single- and multi-process data loading](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading),
		- [automatic memory pinning](https://pytorch.org/docs/stable/data.html#memory-pinning).💡[[What is memory pinning in pytorch]]
	+ **🔖DataSet**
		The most important argument of [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") constructor is [`dataset`](https://pytorch.org/docs/stable/utils.html#module-torch.utils.data.dataset "torch.utils.data.dataset"), which indicates a dataset object to load data from. PyTorch supports two different types of datasets:
		- [map-style datasets](https://pytorch.org/docs/stable/data.html#map-style-datasets),
		- [iterable-style datasets](https://pytorch.org/docs/stable/data.html#iterable-style-datasets).
		
		**Map-style datasets**
		A map-style dataset is one that implements the `__getitem__()` and `__len__()` protocols, and represents a map from (possibly non-integral) indices/keys to data samples.
		
		For example, such a dataset, when accessed with `dataset[idx]`, could read the `idx`-th image and its corresponding label from a folder on the disk.
		
		See [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset "torch.utils.data.Dataset") for more details.

		**Iterable-style datasets**
		An iterable-style dataset is an instance of a subclass of [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset "torch.utils.data.IterableDataset") that implements the `__iter__()` protocol, and represents an iterable over data samples. This type of datasets is particularly suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data.
		
		For example, such a dataset, when called `iter(dataset)`, could return a stream of data reading from a database, a remote server, or even logs generated in real time.
		
		See [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset "torch.utils.data.IterableDataset") for more details. 

## 2. Creating model

To define a neural network in PyTorch, we create a class that inherits from [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We define the layers of the network in the `__init__` function and specify how data will pass through the network in the `forward` function. To accelerate operations in the neural network, we move it to the [accelerator](https://pytorch.org/docs/stable/torch.html#accelerators) such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```



## 3. Optimizing the Model Parameters

To train a model, we need a [loss function](https://pytorch.org/docs/stable/nn.html#loss-functions) and an [optimizer](https://pytorch.org/docs/stable/optim.html).
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```


In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.
```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

We also check the model’s performance against the test dataset to ensure it is learning.
```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

The training process is conducted over several iterations (_epochs_). During each epoch, the model learns parameters to make better predictions. We print the model’s accuracy and loss at each epoch; we’d like to see the accuracy increase and the loss decrease with every epoch.
```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

``` text fold
Epoch 1
-------------------------------
loss: 2.303494  [   64/60000]
loss: 2.294637  [ 6464/60000]
loss: 2.277102  [12864/60000]
loss: 2.269977  [19264/60000]
loss: 2.254234  [25664/60000]
loss: 2.237145  [32064/60000]
loss: 2.231056  [38464/60000]
loss: 2.205036  [44864/60000]
loss: 2.203239  [51264/60000]
loss: 2.170890  [57664/60000]
Test Error:
 Accuracy: 53.9%, Avg loss: 2.168587

Epoch 2
-------------------------------
loss: 2.177784  [   64/60000]
loss: 2.168083  [ 6464/60000]
loss: 2.114908  [12864/60000]
loss: 2.130411  [19264/60000]
loss: 2.087470  [25664/60000]
loss: 2.039667  [32064/60000]
loss: 2.054271  [38464/60000]
loss: 1.985452  [44864/60000]
loss: 1.996019  [51264/60000]
loss: 1.917239  [57664/60000]
Test Error:
 Accuracy: 60.2%, Avg loss: 1.920371
...

Done!
```

## 4. Save models
A common way to save a model is to serialize the internal state dictionary (containing the model parameters).
```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

## 5. Loading models
The process for loading a model includes re-creating the model structure and loading the state dictionary into it.
```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
```
This model can now be used to make predictions.
```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```
Read more about [Saving & Loading your model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).

[^1]: *mainly ref* https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
