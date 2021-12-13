from tsne_torch import TorchTSNE as TSNE

emb_agg = model_good.agg_x.cpu()  # shape (n_samples, d)
emb_agg = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(emb_agg)  # returns shape (n_samples, 2)

emb_r = model_good.x_r.cpu()
emb_r = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(emb_r)  # returns shape (n_samples, 2)

emb_s = model_good.x_s.cpu() 
emb_s = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(emb_s)  # returns shape (n_samples, 2)

emb_e = model_good.x_e.cpu()
emb_e = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(emb_e)  # returns shape (n_samples, 2)

emb_list_good = [emb_agg, emb_r, emb_s, emb_e]

from pandas import Series, DataFrame
result_list = list()

for num, emb in enumerate(emb_list):
    x_value = emb[:,0]
    y_value = emb[:,1]
    label = [num for i in range(30)]
    data = {
        'x': x_value,
        'y': y_value,
        'label' : label
    }
    df = DataFrame(data)
    result_list_good.append(df)
    
result_df = pd.concat([result_list[i] for i in range(4)], ignore_index=True)

result_df.loc[result_df['label'] == 0, 'label'] = 'total'
result_df.loc[result_df['label'] == 1, 'label'] = 'rating'
result_df.loc[result_df['label'] == 2, 'label'] = 'sentiment'
result_df.loc[result_df['label'] == 3, 'label'] = 'emotion'


plt.figure(figsize = (10,6))
sns.set_palette(sns.color_palette("Paired"))
ax = sns.scatterplot(x="x", y="y", hue='label', style='label' , data=result_df, palette='deep', s=50)

# Customize the axes and title
ax.set_title("t-SNE visualization")
ax.set_xlabel("latent_space_x")
ax.set_ylabel("latent_space_y")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()