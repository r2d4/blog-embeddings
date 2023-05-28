# Blog Embeddings

An analysis of two years of daily blogging on [matt-rickard.com](https://matt-rickard.com). 

1. I embedded all my posts using BERT (a transformers model pre-trained on a large corpus of English data). BERT uses 768-dimensional vectors.

2. Then I ran them through t-SNE (t-distributed stochastic neighbor embedding, a fancy way to visualize high-dimensional data by translating them to two dimensions. 

3. Finally, I separated the two-dimensional space into equally sized bins and asked GPT-3.5 to develop a category name for each set of post titles. 