# %%
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR

import os

# %%
os.makedirs("checkpoints/", exist_ok=True)

# %% [markdown]
# ### Load Data

# %%
df = pd.read_csv("./Students_Grading_Dataset.csv")

# %% [markdown]
# ## Manually select Cols (attribute)

# %%
unimportant_attribute = [
    "Student_ID",
    "First_Name",
    "Last_Name",
    "Email",
    "Participation_Score",
]

filtered_df = df.drop(unimportant_attribute, axis=1)
filtered_df

# %%
category_vars = [
    "Gender",
    "Department",
    "Grade",
    "Extracurricular_Activities",
    "Internet_Access_at_Home",
    "Parent_Education_Level",
    "Family_Income_Level",
]
# numerical_score_vars = ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score', 'Stress_Level (1-10)']
numerical_score_vars = [
    "Attendance (%)",
    "Midterm_Score",
    "Final_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Projects_Score",
    "Total_Score",
]
numerical_scalar_vars = list(
    set(filtered_df.columns) - set(category_vars) - set(numerical_score_vars)
)
numerical_scalar_vars

# %% [markdown]
# ## Separate rows with-Nan and without-Nan

# %%
nan_rows = filtered_df.isna().any(axis=1)

# Nan rows
df_nan = filtered_df[nan_rows]
print(f"row with Nan: {df_nan.shape}")

# Complete rows
df_complete = filtered_df[~nan_rows]
print(f"row without Nan: {df_complete.shape}")

# %%
df_train, df_valid, _, _ = train_test_split(
    df_complete, df_complete, test_size=0.3, random_state=0
)

df_train_id = [i for i in range(len(df_train))]
df_valid_id = [i for i in range(len(df_valid))]

print(f"df_train: {df_train.shape}")
print(f"df_valid: {df_valid.shape}")


# %% [markdown]
# ## Preprocessing:
# 1. category to numerical
# 2. max-min norm


# %%
def category_to_numerical(data):
    le = LabelEncoder()
    le.fit(data)
    num_data = le.transform(data)

    return num_data, le


def max_min_norm_score(data, train_params=None, process_type="train"):

    if process_type == "train":
        data_max = 100
        data_min = 0
    else:
        data_max = 100
        data_min = 0

    norm_data = (data - data_min) / (data_max - data_min)

    if process_type == "train":
        return norm_data, data_max, data_min
    else:
        return norm_data


def max_min_norm_scalar(data, train_params=None, process_type="train"):

    if process_type == "train":
        data_max = 10
        data_min = 0
    else:
        data_max = 10
        data_min = 0

    norm_data = (data - data_min) / (data_max - data_min)

    if process_type == "train":
        return norm_data, data_max, data_min
    else:
        return norm_data


def preprocessing(df, train_params=None, process_type="train"):

    new_df = pd.DataFrame()

    if process_type == "train":
        train_params = {}
        category_var_len = {}

    # Category
    for cat_name in category_vars:
        cat_var = df[cat_name]
        if process_type == "train":
            cat_var, le = category_to_numerical(cat_var)
            train_params[f"{cat_name}_le"] = le
            category_var_len[f"{cat_name}"] = len(np.unique(cat_var))
        else:
            cat_var = train_params[f"{cat_name}_le"].transform(cat_var)
        new_df[f"{cat_name}"] = cat_var

    # Numerical score
    for num_name in numerical_score_vars:
        num_var = df[num_name]
        if process_type == "train":
            num_var, data_max, data_min = max_min_norm_score(
                num_var, process_type="train"
            )
            train_params[num_name] = [data_max, data_min]
        else:
            num_var = max_min_norm_score(num_var, train_params, process_type="valid")
        new_df[num_name] = num_var.values

    # Numerical scalar
    for num_name in numerical_scalar_vars:
        num_var = df[num_name]
        num_var = np.log(num_var)
        if process_type == "train":
            num_var, data_max, data_min = max_min_norm_scalar(
                num_var, process_type="train"
            )
            train_params[num_name] = [data_max, data_min]
        else:
            num_var = max_min_norm_scalar(num_var, train_params, process_type="valid")
        new_df[num_name] = num_var.values

    if process_type == "train":
        return new_df, train_params, category_var_len
    else:
        return new_df


# %%
processed_df_train, train_params, category_var_len = preprocessing(
    df_train, process_type="train"
)
# train_params
print(f"category_var_len: {category_var_len}")
print(f"processed_df_train: {processed_df_train.shape}")

cols_name = processed_df_train.columns
processed_df_train.head()

# %%
processed_df_valid = preprocessing(df_valid, train_params, process_type="valid")
print(f"processed_df_valid: {processed_df_valid.shape}")
processed_df_valid.head()

# %%
BATCH_SIZE = 512
# MASK_RATIO = 30


# %%
def masking_helper(data, MASK_RATIO, seed=42):
    np.random.seed(seed)

    rows, cols = data.shape

    # define mask/unmask ratio
    unmask_ratio = ((100 - MASK_RATIO) * cols) // 100
    mask_ratio = cols - unmask_ratio

    # create random index
    # shuff_idx = np.array([np.random.permutation(cols) for _ in range(rows)])
    shuff_idx = np.random.permutation(cols).reshape(1, cols)
    shuff_idx = np.repeat(shuff_idx, rows, axis=0)

    # define mask/unmask idx
    mask_idx = shuff_idx[:, :mask_ratio]
    unmask_idx = shuff_idx[:, mask_ratio:]

    mask_idx.sort(axis=1)
    unmask_idx.sort(axis=1)

    # create new_data (contain unmask cols, but remove mask cols)
    new_data = np.zeros((rows, unmask_ratio))
    for i in range(rows):
        new_data[i] = data[i][unmask_idx[i]]

    return new_data, unmask_idx, mask_idx, unmask_ratio


def masking_fn(data, mask_ratio, seed):
    sample_size = len(data)

    new_data, unmask_idx, mask_idx, unmask_ratio = masking_helper(
        data, mask_ratio, seed
    )

    X = [[], [], [], []]
    Y = []

    for i in range(sample_size):
        X[0].append(new_data[i])  # unmask data
        X[1].append(unmask_idx[i])  # unmask id
        X[2].append(mask_idx[i])  # mask id
        X[3].append(
            np.ones(unmask_ratio)
        )  # len = unmask, serve as random noisee in VAE
        Y.append(
            data[i][list(unmask_idx[i]) + list(mask_idx[i])]
        )  # label (unmask + mask)

    X[0] = torch.tensor(np.array(X[0]))  # unmask data
    X[1] = torch.tensor(np.array(X[1]))  # unmask id
    X[2] = torch.tensor(np.array(X[2]))  # mask id
    X[3] = torch.tensor(np.array(X[3]))  # latent space

    Y = torch.tensor(np.array(Y))

    return X, Y


# %%
def collate_fn_train(data_wtih_type):

    # data
    data = [dat[0] for dat in data_wtih_type]
    data = np.array(data)

    # data type: training/valid
    data_type = data_wtih_type[0][1]

    if data_type == "training":
        seed = np.random.randint(100000, size=1)
        # mask_ratio = np.random.randint(low=10, high=45 + 1, size=1)
        mask_ratio = np.random.randint(low=25, high=40 + 1, size=1)
        mask_ratio = mask_ratio[0]

    else:
        seed = 42
        mask_ratio = 15

    data, label = masking_fn(data, mask_ratio, seed)

    return data, label, mask_ratio


class TableDataset(Dataset):
    def __init__(self, data, data_type="training"):
        self.data = data
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        dat = self.data.iloc[id, :]
        dat = np.array(dat)
        return dat, self.data_type


# %%
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


augmented_processed_df_train = [processed_df_train] * 100
augmented_processed_df_train = pd.concat(augmented_processed_df_train, axis=0)
augmented_processed_df_train = augmented_processed_df_train.sample(frac=1, random_state=42)

print(f"num training data: {augmented_processed_df_train.shape}")

train_dataset = TableDataset(augmented_processed_df_train)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=False,
    collate_fn=collate_fn_train,
)


valid_dataset = TableDataset(processed_df_valid, data_type="valid")
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=False,
    collate_fn=collate_fn_train,
)


# %%
class Head(nn.Module):
    def __init__(self, num_head, encoder_emb_dim, dropout):
        super().__init__()
        self.num_head = num_head
        self.encoder_emb_dim = encoder_emb_dim

        # up_emb_size * 3 for qkv
        self.qkv_fn = nn.Linear(
            self.encoder_emb_dim, self.encoder_emb_dim * 3, bias=False
        )
        self.proj_qkv = nn.Linear(
            self.encoder_emb_dim, self.encoder_emb_dim, bias=False
        )

        # other
        self.att_dropout = nn.Dropout(dropout)

    def softclamp(self, t, value=50.0):
        return (t / value).tanh() * value

    def forward(self, x, iterative):

        batch_size, num_vars, _ = x.shape

        # qkv: up_emb_size * 3
        x = self.qkv_fn(x)
        q, k, v = x.split(self.encoder_emb_dim, dim=2)

        # split head: each shape = [batch_size, num_head, num_vars(seq_len), head_size = 8]
        q = q.view(
            batch_size, num_vars, self.num_head, self.encoder_emb_dim // self.num_head
        ).transpose(1, 2)
        k = k.view(
            batch_size, num_vars, self.num_head, self.encoder_emb_dim // self.num_head
        ).transpose(1, 2)
        v = v.view(
            batch_size, num_vars, self.num_head, self.encoder_emb_dim // self.num_head
        ).transpose(1, 2)

        # attention matrix calculation: [batch_size, num_head, num_vars(seq_len), num_vars]
        att = (q @ k.transpose(-2, -1)) * (
            1 / torch.sqrt(torch.ones([1]).to(device) * k.size(-1))
        )

        # masking the att from unmask to mask (Q_unmask to K_mask),
        # because mask is basically noise, it's meaningless to ask informative vector (Q_unmask) to refer to noise vector (K_mask)
        """
        it's not causal mask
        it looks like:
                unmask, unmaskm, mask  
        unmask  [1,      1,      0] 
        unmask  [1,      1,      0] 
        mask    [1,      1,      1] 
        """
        if iterative:
            mask_id = num_vars - 1
            att_mask = torch.ones_like(att)
            att_mask[:, :, :mask_id, -1] = 0
            att = att.masked_fill(att_mask == 0, float("-inf"))

        # att = self.softclamp(att, 15)

        att = F.softmax(att, dim=-1)
        att = self.att_dropout(att)

        # att matrix * V: [batch_size, num_head, num_vars(seq_len), head_size]
        out = att @ v
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_vars, self.encoder_emb_dim)
        )
        out = self.proj_qkv(out)

        return out


class MLP(nn.Module):

    def __init__(self, transformer_emb_size, dropout):
        super().__init__()
        self.up = nn.Linear(transformer_emb_size, transformer_emb_size * 3)
        self.relu = nn.ReLU()
        self.down = nn.Linear(transformer_emb_size * 3, transformer_emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.down(x)

        return x


class Block(nn.Module):
    def __init__(self, encoder_emb_dim, num_head, dropout):
        super(Block, self).__init__()
        self.ln_head = nn.LayerNorm(encoder_emb_dim)
        self.head = Head(num_head, encoder_emb_dim, dropout)
        self.dropout_head = nn.Dropout(dropout)

        self.ln_mlp = nn.LayerNorm(encoder_emb_dim)
        self.mlp = MLP(encoder_emb_dim, dropout)
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(self, x, iterative):

        # att
        ori_x = x
        x = self.head(x, iterative)
        x = self.dropout_head(x)
        x = self.ln_head(ori_x + x)

        # linear
        ori_x = x
        x = self.mlp(x)
        x = self.dropout_mlp(x)
        x = self.ln_mlp(ori_x + x)

        return x


class Transformer(nn.Module):
    def __init__(self, encoder_emb_dim, layers, num_head, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(encoder_emb_dim, num_head, dropout) for _ in range(layers)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, iterative=False):

        for block in self.blocks:
            x = block(x, iterative)

        return x


# %%
category_vars = [
    "Gender",
    "Department",
    "Grade",
    "Extracurricular_Activities",
    "Internet_Access_at_Home",
    "Parent_Education_Level",
    "Family_Income_Level",
]
numerical_score_vars = [
    "Attendance (%)",
    "Midterm_Score",
    "Final_Score",
    "Assignments_Avg",
    "Quizzes_Avg",
    "Projects_Score",
    "Total_Score",
]
numerical_scalar_vars = list(
    set(filtered_df.columns) - set(category_vars) - set(numerical_score_vars)
)
numerical_scalar_vars


# %%
class Feature_Embed(nn.Module):
    def __init__(
        self,
        cols_name,
        category_vars,
        feature_emb_size,
        num_variables,
        position_emb_dim,
        category_var_len,
        dropout,
    ):
        super().__init__()
        self.cols_name = cols_name
        self.category_vars = category_vars
        self.feature_emb_size = feature_emb_size
        self.category_var_len = category_var_len
        self.hidden_size = feature_emb_size + position_emb_dim

        # Category feature embedding
        self.cat_encoders = {
            "Gender": nn.Embedding(category_var_len["Gender"] + 1, feature_emb_size),
            "Department": nn.Embedding(
                category_var_len["Department"] + 1, feature_emb_size
            ),
            "Grade": nn.Embedding(category_var_len["Grade"] + 1, feature_emb_size),
            "Extracurricular_Activities": nn.Embedding(
                category_var_len["Extracurricular_Activities"] + 1, feature_emb_size
            ),
            "Internet_Access_at_Home": nn.Embedding(
                category_var_len["Internet_Access_at_Home"] + 1, feature_emb_size
            ),
            "Parent_Education_Level": nn.Embedding(
                category_var_len["Parent_Education_Level"] + 1, feature_emb_size
            ),
            "Family_Income_Level": nn.Embedding(
                category_var_len["Family_Income_Level"] + 1, feature_emb_size
            ),
        }

        self.embedding_dropout = nn.Dropout(dropout)

        # self.cat_decoders = {
        #     'Gender': nn.Linear(self.hidden_size, category_var_len['Gender'] + 1, bias=False),
        #     'Department': nn.Linear(self.hidden_size, category_var_len['Department'] + 1, bias=False),
        #     'Grade': nn.Linear(self.hidden_size, category_var_len['Grade'] + 1, bias=False),
        #     'Extracurricular_Activities': nn.Linear(self.hidden_size, category_var_len['Extracurricular_Activities'] + 1, bias=False),
        #     'Internet_Access_at_Home': nn.Linear(self.hidden_size, category_var_len['Internet_Access_at_Home'] + 1, bias=False),
        #     'Parent_Education_Level': nn.Linear(self.hidden_size, category_var_len['Parent_Education_Level'] + 1, bias=False),
        #     'Family_Income_Level': nn.Linear(self.hidden_size, category_var_len['Family_Income_Level'] + 1, bias=False),
        # }
        self.LayerNorm = nn.LayerNorm(self.hidden_size)

        self.cat_decoders = {
            "Gender": nn.Linear(
                self.hidden_size, category_var_len["Gender"], bias=False
            ),
            "Department": nn.Linear(
                self.hidden_size, category_var_len["Department"], bias=False
            ),
            "Grade": nn.Linear(self.hidden_size, category_var_len["Grade"], bias=False),
            "Extracurricular_Activities": nn.Linear(
                self.hidden_size,
                category_var_len["Extracurricular_Activities"],
                bias=False,
            ),
            "Internet_Access_at_Home": nn.Linear(
                self.hidden_size,
                category_var_len["Internet_Access_at_Home"],
                bias=False,
            ),
            "Parent_Education_Level": nn.Linear(
                self.hidden_size, category_var_len["Parent_Education_Level"], bias=False
            ),
            "Family_Income_Level": nn.Linear(
                self.hidden_size, category_var_len["Family_Income_Level"], bias=False
            ),
        }

        # numerical encoder
        self.numerical_encoder = nn.Linear(1, feature_emb_size, bias=False)

        # numerical decoder
        self.numerical_decoder1 = nn.Linear(
            self.hidden_size, self.hidden_size * 3, bias=False
        )
        self.numerical_decoder2 = nn.Linear(
            self.hidden_size * 3, self.hidden_size, bias=False
        )
        # self.numerical_decoder3 = nn.Linear(self.hidden_size * 4, self.hidden_size * 2, bias = False)
        # self.numerical_decoder4 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias = False)
        self.numerical_decoder5 = nn.Linear(self.hidden_size, 1, bias=False)

        self.relu = nn.ReLU()

        self.cols_id_name = {}
        count = 0
        for col in self.cols_name:
            self.cols_id_name[count] = col
            count += 1

        # positional embedding
        self.position_emb_dim = position_emb_dim
        self.pos_emb = nn.Embedding(num_variables, position_emb_dim)

    def encode(self, unmasked_data, unmasked_idx, masked_idx):

        batch_size, num_unmask_cols = unmasked_data.shape
        _, num_mask_cols = masked_idx.shape
        device = unmasked_data.device

        # position info
        unmasked_pos_info = self.pos_emb(unmasked_idx.long()).float()
        masked_pos_info = self.pos_emb(masked_idx.long()).float()

        """ Feature Encoding for Unmask """

        # iterate every col in unmask
        # for category, do nn.Embedding
        # for numerical, repeat feature_emb_size times
        unmasked_emb = []
        for c in range(num_unmask_cols):
            unmask_attribute_value = unmasked_data[:, c]

            # make sure unmask_attribute_id is unique along one col dimension
            # use the unique id to see the choosen attribute is categorial or numerical
            unmask_attribute_id = unmasked_idx[:, c]
            assert (
                len(torch.unique(unmask_attribute_id)) == 1
            ), "unmask_attribute_id in one small batch has to be the same"
            unmask_attribute_id = torch.unique(unmask_attribute_id).item()

            # this col is categorial
            if self.cols_id_name[unmask_attribute_id] in self.category_vars:
                encoder = self.cat_encoders[self.cols_id_name[unmask_attribute_id]].to(
                    device
                )
                unmask_attribute_emb = encoder(unmask_attribute_value.long()).float()
                unmask_attribute_emb = self.embedding_dropout(unmask_attribute_emb)
                unmasked_emb.append(unmask_attribute_emb)

            # this col is numerical
            else:
                unmask_attribute_value = torch.unsqueeze(unmask_attribute_value, dim=1)
                unmask_attribute_emb = self.numerical_encoder(unmask_attribute_value)
                # print(f"unmask_attribute_emb: {unmask_attribute_emb.shape}")
                # unmask_attribute_value = torch.unsqueeze(unmask_attribute_value, dim = 1)
                # unmask_attribute_emb = unmask_attribute_value.repeat(1, self.feature_emb_size)
                # print(f"unmask_attribute_emb: {unmask_attribute_emb.shape}")
                # raise
                unmasked_emb.append(unmask_attribute_emb)

        unmasked_emb = torch.stack(unmasked_emb, dim=1)

        """ Positional Encoding for Unmask """
        unmasked_emb = torch.cat(
            [unmasked_emb, unmasked_pos_info], dim=2
        )  # (batch_size, num_unmask_vars, feature_emb_size + pos_emb_size)

        """ Feature Encoding for Mask """
        # create noise for categorial and numerical seperately
        masked_emb = []
        for c in range(num_mask_cols):
            mask_attribute_id = masked_idx[:, c]
            assert (
                len(torch.unique(mask_attribute_id)) == 1
            ), "mask_attribute_id in one small batch has to be the same"
            mask_attribute_id = torch.unique(mask_attribute_id).item()

            # this col is categorial
            if self.cols_id_name[mask_attribute_id] in self.category_vars:

                # get col name
                col_name = self.cols_id_name[mask_attribute_id]
                encoder = self.cat_encoders[col_name].to(device)

                # create categorial mask id
                category_mask_id = self.category_var_len[col_name]
                categorial_latent = torch.ones(batch_size).to(device) * category_mask_id
                categorial_emb = encoder(categorial_latent.long()).float()
                categorial_emb = self.embedding_dropout(categorial_emb)
                masked_emb.append(categorial_emb)

            # this col is numerical
            else:
                # create rand(0-1) as numerical latent
                # numerical_latent = torch.ones(batch_size).to(device) * torch.rand(1).to(device)
                numerical_latent = torch.rand(batch_size).to(
                    device
                )  # * torch.rand(1).to(device)
                # numerical_latent = torch.unsqueeze(numerical_latent, dim = 1)
                # numerical_emb = numerical_latent.repeat(1, self.feature_emb_size)

                numerical_latent = torch.unsqueeze(numerical_latent, dim=1)
                numerical_emb = self.numerical_encoder(numerical_latent)

                masked_emb.append(numerical_emb)

        masked_emb = torch.stack(masked_emb, dim=1)

        """ Positional Encoding for Mask """
        masked_emb = torch.cat(
            [masked_emb, masked_pos_info], dim=2
        )  # (batch_size, num_mask_vars, feature_emb_size + pos_emb_size)

        return unmasked_emb, masked_emb

    def decode(self, all_emb, unmask_mask_id):

        _, num_cols, _ = all_emb.shape
        device = all_emb.device

        """ decode for unmask and mask """
        categorial_preds = []
        numerical_preds = []

        categorial_col_name = []
        numerical_col_name = []

        for c in range(num_cols):
            # attribute value
            attribute_value = all_emb[:, c, :]

            # attribute id
            attribute_id = unmask_mask_id[:, c]
            assert (
                len(torch.unique(attribute_id)) == 1
            ), "attribute_id in one small batch has to be the same"
            attribute_id = torch.unique(attribute_id).item()

            # this col is categorial
            if self.cols_id_name[attribute_id] in self.category_vars:
                col_name = self.cols_id_name[attribute_id]
                decoder = self.cat_decoders[col_name].to(device)

                attribute_value = self.LayerNorm(attribute_value)

                cat_pred = decoder(attribute_value)

                categorial_preds.append(cat_pred)
                categorial_col_name.append(col_name)

            # this col is numerical
            else:
                col_name = self.cols_id_name[attribute_id]
                numerical_col_name.append(col_name)
                # attribute_value = self.LayerNorm(attribute_value)

                numerical_preds.append(attribute_value)

        # decode for numerical
        numerical_preds = torch.stack(numerical_preds, dim=1)

        ori_numerical_preds = numerical_preds

        numerical_preds = self.numerical_decoder1(numerical_preds).to(device)
        numerical_preds = self.relu(numerical_preds)

        numerical_preds = self.numerical_decoder2(numerical_preds).to(device)
        numerical_preds = self.relu(numerical_preds)

        # residual connection
        numerical_preds = numerical_preds + ori_numerical_preds
        numerical_preds = self.numerical_decoder5(numerical_preds).to(device)

        numerical_preds = torch.squeeze(numerical_preds)

        return (
            categorial_preds,
            numerical_preds,
            categorial_col_name,
            numerical_col_name,
            self.cols_id_name,
        )

    def reorder_label_fn(self, label, unmask_mask_id):

        _, num_cols = label.shape
        _ = label.device

        """ reorder for unmask and mask """
        categorial_labels = []
        numerical_labels = []

        for c in range(num_cols):
            # attribute value
            attribute_value = label[:, c]

            # attribute id
            attribute_id = unmask_mask_id[:, c]
            assert (
                len(torch.unique(attribute_id)) == 1
            ), "attribute_id in one small batch has to be the same"
            attribute_id = torch.unique(attribute_id).item()

            # this col is categorial
            if self.cols_id_name[attribute_id] in self.category_vars:
                categorial_labels.append(attribute_value)

            # this col is numerical
            else:
                numerical_labels.append(attribute_value)

        numerical_labels = torch.stack(numerical_labels, dim=1)

        return categorial_labels, numerical_labels


# %%
class tableMET(nn.Module):
    def __init__(
        self,
        position_emb_dim,
        layers_encode,
        layers_decode,
        layers_iterative_transformer,
        num_head,
        num_variables,
        cols_name,
        category_vars,
        feature_emb_size,
        category_var_len,
        dropout,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.cols_name = cols_name

        # feature embedding
        self.feature_emb_fn = Feature_Embed(
            cols_name,
            category_vars,
            feature_emb_size,
            num_variables,
            position_emb_dim,
            category_var_len,
            dropout,
        )

        # encoder and decoder
        self.transformer_encode = Transformer(
            position_emb_dim + feature_emb_size, layers_encode, num_head, dropout
        )
        self.transformer_decode = Transformer(
            position_emb_dim + feature_emb_size, layers_decode, num_head, dropout
        )

        # iterative transformer decoding
        self.iterative_transformer_decode = Transformer(
            position_emb_dim + feature_emb_size,
            layers_iterative_transformer,
            num_head,
            dropout,
        )

    def forward(self, unmasked_data, unmasked_idx, masked_idx, label, mask_ratio):
        self.unmask_ratio = ((100 - mask_ratio) * self.num_variables) // 100
        self.mask_ratio = (
            self.num_variables - ((100 - mask_ratio) * self.num_variables) // 100
        )

        """ feature & position encoding for Unmask and Mask """
        unmask_emb, mask_emb = self.feature_emb_fn.encode(
            unmasked_data, unmasked_idx, masked_idx
        )

        # transformer (encoder) only for unmasked part
        ori_unmask_emb = unmask_emb
        unmask_emb = self.transformer_encode(unmask_emb)
        unmask_emb = unmask_emb + ori_unmask_emb

        # iterative transformer decoding
        mask_emb_iter = []
        for mask_id in range(self.mask_ratio):
            maske_e = torch.unsqueeze(mask_emb[:, mask_id, :], dim=1)
            unmask_mask_emb = torch.cat([unmask_emb, maske_e], dim=1)
            unmask_mask_emb = self.iterative_transformer_decode(
                unmask_mask_emb, iterative=True
            )
            maske_e = torch.unsqueeze(unmask_mask_emb[:, -1, :], dim=1)
            mask_emb_iter.append(maske_e)

        mask_emb_iter = torch.cat(mask_emb_iter, dim=1)

        # concat unmask_emb and mask_emb
        all_emb = torch.cat([unmask_emb, mask_emb_iter], dim=1)

        # transformer (decoder) for both unmasked and masked
        ori_all_emb = all_emb
        all_emb = self.transformer_decode(all_emb)
        all_emb = all_emb + ori_all_emb

        """ feature & position decoding for Unmask and Mask """
        unmask_mask_id = torch.cat([unmasked_idx, masked_idx], dim=1)
        (
            categorial_preds,
            numerical_preds,
            categorial_col_name,
            numerical_col_name,
            _,
        ) = self.feature_emb_fn.decode(all_emb, unmask_mask_id)

        """ reorder label: make label' variable order align as pred """
        categorial_labels, numerical_labels = self.feature_emb_fn.reorder_label_fn(
            label, unmask_mask_id
        )

        """only calculate MASK token loss"""
        mask_col_names = self.cols_name[masked_idx[0].detach().cpu().numpy().tolist()]

        mask_categorial_preds = []
        mask_categorial_labels = []

        for i, col_name in enumerate(categorial_col_name):
            if col_name in mask_col_names:
                categorial_pred = categorial_preds[i]
                categorial_label = categorial_labels[i]
                mask_categorial_preds.append(categorial_pred)
                mask_categorial_labels.append(categorial_label)

        mask_numerical_preds = []
        mask_numerical_labels = []

        for i, col_name in enumerate(numerical_col_name):
            if col_name in mask_col_names:
                numerical_pred = torch.unsqueeze(numerical_preds[:, i], dim=1)
                numerical_label = torch.unsqueeze(numerical_labels[:, i], dim=1)
                mask_numerical_preds.append(numerical_pred)
                mask_numerical_labels.append(numerical_label)

        if len(mask_numerical_preds) > 0:
            mask_numerical_preds = torch.cat(mask_numerical_preds, dim=1)
            mask_numerical_labels = torch.cat(mask_numerical_labels, dim=1)
        else:
            mask_numerical_preds = []
            mask_numerical_labels = []

        return (
            mask_categorial_preds,
            mask_numerical_preds,
            mask_categorial_labels,
            mask_numerical_labels,
            categorial_col_name,
            numerical_col_name,
        )


# %%
def linear_warmup_decay_lr(lr_init, lr_final, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps
        else:
            progress = (current_step - num_warmup_steps) / (
                num_training_steps - num_warmup_steps
            )
            return (1 - progress) * (1 - lr_final / lr_init) + (lr_final / lr_init)

    return lr_lambda


# %%
LEARNING_RATE = 1e-4
EPOCHS = 600
layers_encode = 12
layers_iterative_transformer = 8
layers_decode = 1
num_head = 2
dropout = 0.3
num_variables = 18
cols_name = processed_df_train.columns
feature_emb_size = 8
position_emb_dim = 256 - 8
category_var_len = category_var_len

is_power_of_two = lambda n: n > 0 and (n & (n - 1)) == 0
assert is_power_of_two(
    position_emb_dim + feature_emb_size
), "position_emb_dim + feature_emb_size should be power term of 2"

model = tableMET(
    position_emb_dim,
    layers_encode,
    layers_decode,
    layers_iterative_transformer,
    num_head,
    num_variables,
    cols_name,
    category_vars,
    feature_emb_size,
    category_var_len,
    dropout,
).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-6, weight_decay=1e-3
)

# num_warmup_steps = int(EPOCHS * 0.05)
num_warmup_steps = 10
scheduler = LambdaLR(
    optimizer,
    lr_lambda=linear_warmup_decay_lr(
        lr_init=LEARNING_RATE,
        lr_final=LEARNING_RATE * 1e-2,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=EPOCHS,
    ),
)


# %%
MSE_loss_fn = nn.MSELoss()
CE_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


def loss_fn(categorial_pred, numerical_pred, categorial_label, numerical_label):

    # there exist MASK numerical
    if isinstance(numerical_pred, torch.Tensor):
        _, num_numerical = numerical_pred.shape
    else:
        num_numerical = 0

    # calcualte amount of categorial
    if len(categorial_pred) > 0:
        num_category = len(categorial_pred)
    else:
        num_category = 0

    ratio_numerical = num_numerical / (num_numerical + num_category)
    ratio_category = 1 / (num_numerical + num_category)

    total_loss = torch.zeros(1).to(device)
    mse_loss = torch.zeros(1).to(device)
    
    if num_numerical > 0:
        mse_loss = MSE_loss_fn(numerical_pred, numerical_label)
        mse_loss = mse_loss * 1
        total_loss += mse_loss * ratio_numerical

    if num_category > 0:
        for i in range(num_category):
            pred = categorial_pred[i]
            label = categorial_label[i]
            loss = CE_loss_fn(pred, label.long())
            total_loss += loss * ratio_category

    return (
        total_loss,
        mse_loss,
        (total_loss - mse_loss).item(),
        num_numerical,
        num_category,
    )


# %%
train_LOSS = []
valid_LOSS = []

for epoch in tqdm(range(EPOCHS), desc="iterate epoch"):

    losses = []
    mse_losses = []
    ce_losses = []

    val_losses = []
    val_mse_losses = []
    val_ce_losses = []

    model.train()
    for data, label, mask_ratio in train_dataloader:
        unmask_ratio = ((100 - mask_ratio) * num_variables) // 100

        unmasked_data = data[0].float().to(device)
        unmasked_idx = data[1].long().to(device)
        masked_idx = data[2].long().to(device)
        latent = data[3].float().to(device)
        label = label.float().to(device)

        (
            train_categorial_pred,
            train_numerical_pred,
            train_categorial_label,
            train_numerical_label,
            categorial_col_name,
            numerical_col_name,
        ) = model(unmasked_data, unmasked_idx, masked_idx, label, mask_ratio)

        loss, mse_loss, ce_loss, num_numerical, num_category = loss_fn(
            train_categorial_pred,
            train_numerical_pred,
            train_categorial_label,
            train_numerical_label,
        )

        losses.append(loss.item())
        mse_losses.append(mse_loss.item())
        ce_losses.append(ce_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    losses = np.mean(losses)
    mse_losses = np.mean(mse_losses)
    ce_losses = np.mean(ce_losses)
    train_LOSS.append(losses)

    if epoch % 5 == 0:
        print(
            f"TRAINING, total variables are {num_variables}: first {unmask_ratio} unmasked, latter {num_variables - unmask_ratio}"
        )
        print(
            f"epoch: {epoch}, loss: {losses}, mse: {mse_losses}, ce: {ce_losses}, num_numerical: {num_numerical}, num_category: {num_category}"
        )

    with torch.no_grad():
        model.eval()
        for data, label, mask_ratio in valid_dataloader:
            unmask_ratio = ((100 - mask_ratio) * num_variables) // 100

            unmasked_data = data[0].float().to(device)
            unmasked_idx = data[1].long().to(device)
            masked_idx = data[2].long().to(device)
            latent = data[3].float().to(device)
            label = label.float().to(device)

            (
                categorial_pred,
                numerical_pred,
                categorial_label,
                numerical_label,
                categorial_col_name,
                numerical_col_name,
            ) = model(unmasked_data, unmasked_idx, masked_idx, label, mask_ratio)

            loss, mse_loss, ce_loss, num_numerical, num_category = loss_fn(
                categorial_pred, numerical_pred, categorial_label, numerical_label
            )

            val_losses.append(loss.item())
            val_mse_losses.append(mse_loss.item())
            val_ce_losses.append(ce_loss)

    val_losses = np.mean(val_losses)
    val_mse_losses = np.mean(val_mse_losses)
    val_ce_losses = np.mean(val_ce_losses)
    valid_LOSS.append(val_losses)

    if epoch % 5 == 0:
        print(
            f"VALID, total variables are {num_variables}: first {unmask_ratio} unmasked, latter {num_variables - unmask_ratio}"
        )
        print(
            f"epoch: {epoch}, val_loss: {val_losses}, val_mse: {val_mse_losses}, val_ce_losses: {val_ce_losses}, num_numerical: {num_numerical}, num_category: {num_category}"
        )
        print()

# model.save_pretrained("checkpoints/FT_MET")

# %%
plt.plot(range(len(train_LOSS)), train_LOSS, color="blue")
plt.plot(range(len(valid_LOSS)), valid_LOSS, color="red")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# # Inference


# %%
def reverse_norm(
    df, mask_category_vars, mask_numerical_score_vars, mask_numerical_scalar_vars
):

    new_df = pd.DataFrame()

    # Category
    for cat_name in mask_category_vars:
        cat_var = df[cat_name]
        new_df[f"{cat_name}"] = cat_var

    # Numerical score
    for num_name in mask_numerical_score_vars:
        num_var = df[num_name]

        # reverse (0 - 100) min max norm
        num_var = (num_var * 100) + 0

        new_df[num_name] = num_var.values

    # Numerical scalar
    for num_name in mask_numerical_scalar_vars:
        num_var = df[num_name]

        # reverse (0 - 10) min max norm
        num_var = (num_var * 10) + 0
        # reverse
        num_var = np.exp(num_var)
        new_df[num_name] = num_var.values

    return new_df


# %%
select_sample_id = 2


# %%
mask_col_names = cols_name[masked_idx[0].detach().cpu().numpy().tolist()]
mask_col_names
print(f"mask_col_names: {mask_col_names}")
print()

print(f"category_vars: {category_vars}")
print(f"numerical_score_vars: {numerical_score_vars}")
print(f"numerical_scalar_vars: {numerical_scalar_vars}")
print()

mask_category_vars = []
for c in mask_col_names:
    if c in category_vars:
        mask_category_vars.append(c)

mask_numerical_score_vars = []
for c in mask_col_names:
    if c in numerical_score_vars:
        mask_numerical_score_vars.append(c)

mask_numerical_scalar_vars = []
for c in mask_col_names:
    if c in numerical_scalar_vars:
        mask_numerical_scalar_vars.append(c)


print(f"mask_category_vars: {mask_category_vars}")
print(f"mask_numerical_score_vars: {mask_numerical_score_vars}")
print(f"mask_numerical_scalar_vars: {mask_numerical_scalar_vars}")


# %%
mask_col_names = cols_name[masked_idx[0].detach().cpu().numpy().tolist()]

categorial_pred_one = []
for pred in categorial_pred:
    argmax_pred = torch.argmax(pred, dim=1)
    categorial_pred_one.append(argmax_pred)
categorial_pred_one = torch.stack(categorial_pred_one, dim=1)
categorial_pred_one = np.array(categorial_pred_one[select_sample_id].detach().cpu())

categorial_label_one = torch.stack(categorial_label, dim=1)

categorial_label_one = np.array(categorial_label_one[select_sample_id].detach().cpu())

numerical_pred_one = np.array(numerical_pred[select_sample_id].detach().cpu())
numerical_label_one = np.array(numerical_label[select_sample_id].detach().cpu())

# %%
pred = np.concat([categorial_pred_one, numerical_pred_one], axis=0)
label = np.concat([categorial_label_one, numerical_label_one], axis=0)

res = pd.DataFrame([pred, label], columns=mask_col_names)
reverse_res = reverse_norm(
    res, mask_category_vars, mask_numerical_score_vars, mask_numerical_scalar_vars
)


# %%
reverse_res

# %%
for select_sample_id in range(len(numerical_pred)):

    mask_col_names = cols_name[masked_idx[0].detach().cpu().numpy().tolist()]
    mask_category_vars = []
    for c in mask_col_names:
        if c in category_vars:
            mask_category_vars.append(c)

    mask_numerical_score_vars = []
    for c in mask_col_names:
        if c in numerical_score_vars:
            mask_numerical_score_vars.append(c)

    mask_numerical_scalar_vars = []
    for c in mask_col_names:
        if c in numerical_scalar_vars:
            mask_numerical_scalar_vars.append(c)

    mask_col_names = cols_name[masked_idx[0].detach().cpu().numpy().tolist()]

    categorial_pred_one = []
    for pred in categorial_pred:
        argmax_pred = torch.argmax(pred, dim=1)
        categorial_pred_one.append(argmax_pred)
    categorial_pred_one = torch.stack(categorial_pred_one, dim=1)
    categorial_pred_one = np.array(categorial_pred_one[select_sample_id].detach().cpu())

    categorial_label_one = torch.stack(categorial_label, dim=1)
    categorial_label_one = np.array(
        categorial_label_one[select_sample_id].detach().cpu()
    )

    numerical_pred_one = np.array(numerical_pred[select_sample_id].detach().cpu())
    numerical_label_one = np.array(numerical_label[select_sample_id].detach().cpu())

    pred = np.concat([categorial_pred_one, numerical_pred_one], axis=0)
    label = np.concat([categorial_label_one, numerical_label_one], axis=0)

    res = pd.DataFrame([pred, label], columns=mask_col_names)
    reverse_res = reverse_norm(
        res, mask_category_vars, mask_numerical_score_vars, mask_numerical_scalar_vars
    )

    print(f"reverse_res: {reverse_res}")
    print()

# %%

# print(f"categorial_preds: {len(categorial_preds)}")
# print(f"numerical_preds: {numerical_preds.shape}")
# print(f"categorial_col_name: {categorial_col_name}")
# print(f"numerical_col_name: {numerical_col_name}")
# print(f"cols_id_name: {cols_id_name}")
# print(f"categorial_labels: {len(categorial_labels)}")
# print(f"numerical_labels: {numerical_labels.shape}")
# print(f"masked_idx: {masked_idx}")

# %%

# Unmask col_name: Gender
# Unmask col_name: Department
# Unmask col_name: Extracurricular_Activities
# Unmask col_name: Internet_Access_at_Home
# Unmask col_name: Family_Income_Level
# mask col_name: Grade
# mask col_name: Parent_Education_Level
# iterate epoch:   0%|          | 0/3000 [00:00<?, ?it/s]
# categorial_preds: 7
# numerical_preds: torch.Size([2, 11])
# categorial_col_name: ['Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home', 'Family_Income_Level', 'Grade', 'Parent_Education_Level']
# numerical_col_name: ['Attendance (%)', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Total_Score', 'Age', 'Stress_Level (1-10)', 'Midterm_Score', 'Projects_Score', 'Sleep_Hours_per_Night', 'Study_Hours_per_Week']
# cols_id_name: {0: 'Gender', 1: 'Department', 2: 'Grade', 3: 'Extracurricular_Activities', 4: 'Internet_Access_at_Home', 5: 'Parent_Education_Level', 6: 'Family_Income_Level', 7: 'Attendance (%)', 8: 'Midterm_Score', 9: 'Final_Score', 10: 'Assignments_Avg', 11: 'Quizzes_Avg', 12: 'Projects_Score', 13: 'Total_Score', 14: 'Sleep_Hours_per_Night', 15: 'Age', 16: 'Study_Hours_per_Week', 17: 'Stress_Level (1-10)'}
# categorial_labels: 7
# numerical_labels: torch.Size([2, 11])
# masked_idx: tensor([[ 2,  5,  8, 12, 14, 16],
#         [ 2,  5,  8, 12, 14, 16]], device='cuda:0')
# mask_col_names: Index(['Grade', 'Parent_Education_Level', 'Midterm_Score', 'Projects_Score',
#     'Sleep_Hours_per_Night', 'Study_Hours_per_Week'],
#     dtype='object')
# cols_name: Index(['Gender', 'Department', 'Grade', 'Extracurricular_Activities',
#     'Internet_Access_at_Home', 'Parent_Education_Level',
#     'Family_Income_Level', 'Attendance (%)', 'Midterm_Score', 'Final_Score',
#     'Assignments_Avg', 'Quizzes_Avg', 'Projects_Score', 'Total_Score',
#     'Sleep_Hours_per_Night', 'Age', 'Study_Hours_per_Week',
#     'Stress_Level (1-10)'],
#     dtype='object')

# %%


# %%
