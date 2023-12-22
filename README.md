# CoRTEx
Publish the model for the paper "CoRTEx: Contrastive Learning for Representing Terms via Explanations with Applications on Constructing Biomedical Knowledge Graphs"

Because of license of UMLS, we cannot share our training data. So we only provide our model and how to use our model. If you need the training details please contact yinghy22@mails.tsinghua.edu.cn

Usage:
1. Download the model from (https://cloud.tsinghua.edu.cn/f/def076f2f5d14fcc8e89/)
2. run the code below, or you can go to the original file to change the codes.

```python
from generate_faiss_index import get_instructor_embed
model= torch.load(ori_Instructor_path).to(device)
get_instructor_embed(phrase_list, model, batch_size=128)
```
Here the "phrase_list" is a python list containing the terms you want to encode.

TODO: create the environment
