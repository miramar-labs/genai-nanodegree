# Miscellaneous Notes

✅ Major Types of Layers - PyTorch Cheat Sheet

Different layers let neural networks extract features, reduce dimensions, remember sequences, and learn complex relationships for a variety of data types.

| **Layer Type**                 | **Use Case** | **PyTorch Class / Function** | **Example**                                         |
| ------------------------------ | -------------| ---------------------------- | --------------------------------------------------- |
| **Fully Connected (Dense)**    | General learning, classification | `nn.Linear`                  | `nn.Linear(128, 64)`                                |
| **Convolutional (1D)**         |Images, spatial data| `nn.Conv1d`                  | `nn.Conv1d(16, 32, kernel_size=3)`                  |
| **Convolutional (2D)**         |Images, spatial data| `nn.Conv2d`                  | `nn.Conv2d(3, 16, kernel_size=3)`                   |
| **Convolutional (3D)**         |Images, spatial data| `nn.Conv3d`                  | `nn.Conv3d(1, 8, kernel_size=3)`                    |
| **Transpose Convolution (2D)** |Images, spatial data| `nn.ConvTranspose2d`         | `nn.ConvTranspose2d(16, 3, kernel_size=3)`          |
| **Max Pooling (1D)**           |Downsampling, summarizing features| `nn.MaxPool1d`               | `nn.MaxPool1d(kernel_size=2)`                       |
| **Max Pooling (2D)**           |Downsampling, summarizing features| `nn.MaxPool2d`               | `nn.MaxPool2d(kernel_size=2)`                       |
| **Average Pooling (2D)**       |Downsampling, summarizing features| `nn.AvgPool2d`               | `nn.AvgPool2d(kernel_size=2)`                       |
| **Global Average Pooling**     |Downsampling, summarizing features| `nn.AdaptiveAvgPool2d`       | `nn.AdaptiveAvgPool2d((1,1))`                       |
| **Flatten Layer**              |Shape conversion before dense layers| `nn.Flatten`                 | `nn.Flatten()`                                      |
| **ReLU Activation**            |Non-linearity| `nn.ReLU`                    | `nn.ReLU()`                                         |
| **Sigmoid Activation**         |Non-linearity| `nn.Sigmoid`                 | `nn.Sigmoid()`                                      |
| **Tanh Activation**            |Non-linearity| `nn.Tanh`                    | `nn.Tanh()`                                         |
| **Leaky ReLU**                 |Non-linearity| `nn.LeakyReLU`               | `nn.LeakyReLU(0.1)`                                 |
| **Softmax Activation**         |Non-linearity| `nn.Softmax`                 | `nn.Softmax(dim=1)`                                 |
| **Dropout**                    |Prevent overfitting| `nn.Dropout`                 | `nn.Dropout(p=0.5)`                                 |
| **Batch Normalization (1D)**   |Stabilize training| `nn.BatchNorm1d`             | `nn.BatchNorm1d(64)`                                |
| **Batch Normalization (2D)**   |Stabilize training| `nn.BatchNorm2d`             | `nn.BatchNorm2d(32)`                                |
| **Layer Normalization**        |Stabilize training| `nn.LayerNorm`               | `nn.LayerNorm(128)`                                 |
| **Group Normalization**        |Stabilize training| `nn.GroupNorm`               | `nn.GroupNorm(num_groups=8, num_channels=32)`       |
| **Embedding Layer**            |NLP, categorical variables| `nn.Embedding`               | `nn.Embedding(10000, 300)`                          |
| **RNN (vanilla)**              |Sequences, time series, text| `nn.RNN`                     | `nn.RNN(128, 64)`                                   |
| **LSTM**                       |Sequences, time series, text| `nn.LSTM`                    | `nn.LSTM(128, 64)`                                  |
| **GRU**                        |Sequences, time series, text| `nn.GRU`                     | `nn.GRU(128, 64)`                                   |
| **Attention (Multi-head)**     |Focus on important parts of input| `nn.MultiheadAttention`      | `nn.MultiheadAttention(embed_dim=512, num_heads=8)` |
| **Transformer Encoder**        |NLP,TxtCls,Sentence Embeddings,NER,QA| `nn.TransformerEncoder`      | `nn.TransformerEncoder(...)`                        |
| **Transformer Decoder**        |TxtxGen,MT,TxtSum,CodeGen| `nn.TransformerDecoder`      | `nn.TransformerDecoder(...)`                        |
| **Transformer Block**          |NLP,MT,TxtSum,TxtCls,QA,NER,ViT,MML,Speech,Audio,TS| `nn.Transformer`             | `nn.Transformer(...)`                               |
