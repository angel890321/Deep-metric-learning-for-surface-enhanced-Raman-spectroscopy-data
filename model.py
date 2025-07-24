import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class RamanNet(nn.Module):
    def __init__(self, w_len=50, n_windows=25,n_classes=30):
        super(RamanNet, self).__init__()
        self.n_windows = n_windows
        self.feature_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(w_len, 25), nn.BatchNorm1d(25), nn.LeakyReLU())
                for _ in range(n_windows)
            ]
        )

        self.embedding = nn.Sequential(
            nn.Dropout(0.50),
            nn.Linear(25 * n_windows, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.40),
            nn.Linear(512, 256),
        )
        

        self.final = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
        )

        self.classification = nn.Linear(256,n_classes )

    def forward(self, x):
        x = x.view(x.shape[0], 14, 50)
        features = [self.feature_layers[i](x[:, i]) for i in range(self.n_windows)]
        combined = torch.cat(features, dim=1)
        combined = self.embedding(combined)
        emb = F.normalize(combined, dim=1, p=2)
        final = self.final(combined)
        out = self.classification(final)
        return out,emb

class Variant_LeNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Variant_LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=21,
            ),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=11,
            ),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
            ),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.embedding = nn.Sequential(
            nn.Linear(in_features=2752, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=out_channels),
            nn.BatchNorm1d(num_features=out_channels),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = conv3.view(conv3.size(0), -1)
        out1 = self.embedding(conv4)
        emb = F.normalize(out1, dim=1, p=2)
        out2 = self.dense2(out1)
        return out2,emb
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

       # Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
                               stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
                               stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, hidden_sizes, num_blocks, input_dim=1000, in_channels=64, n_classes=30, is_norm=True
    ):
        super(ResNet, self).__init__()
        assert len(num_blocks) == len(hidden_sizes)
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        # Flexible number of residual encoding layers
        layers = []
        strides = [1] + [2] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                                           stride=strides[idx]))
        self.encoder = nn.Sequential(*layers)

        self.z_dim = self._get_encoding_size()
        self.embedding = nn.Linear(self.z_dim, 256)  # 加入 embedding 層
        self.classifier = nn.Linear(256, self.n_classes)
        self.is_norm = is_norm

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        output = torch.div(input, norm.view(-1, 1).expand_as(input))
        return output.view(input_size)

    def encode(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        z = self.encode(x)
        x = self.embedding(z)           # 對應 self.model.embedding(x)
        if self.is_norm:
            emb = self.l2_norm(x)   
        out = self.classifier(emb)        # 最終分類
        return out ,emb 


    def _make_layer(self, out_channels, num_blocks, stride=50):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels, stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _get_encoding_size(self):
        """
        Returns the dimension of the encoded input.
        """
        temp = Variable(torch.rand(1, 1, self.input_dim))
        z = self.encode(temp)
        z_dim = z.data.size(1)
        return z_dim
    
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.embedding = nn.Linear(dim, 256)
        self.mlp_head = nn.Linear(256, num_classes)
    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        return _output.view(input_size)

    def forward(self, img):
        img = img.reshape((img.shape[0],1, 1, -1))
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x_latent = self.to_latent(x)
        x_embed = self.embedding(x_latent)  # <--- embedding 層
        x_embed = self.l2_norm(x_embed) 
        out = self.mlp_head(x_embed)
        return out,x_embed
