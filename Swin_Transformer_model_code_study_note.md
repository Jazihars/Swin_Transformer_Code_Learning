# Swin Transformer模型代码学习笔记
在这份笔记中，我将逐行分析Swin Transformer模型设计部分的代码。弄懂Swin Transformer的模型究竟是如何实现的。

## Swin Transformer模型的初始化
在Swin Transformer的训练脚本`/Swin-Transformer/main.py`里，有一个`def main(config):`函数，它的部分代码如下：
``` python
def main(config):
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
```
这段代码就是构造Swin Transformer模型的代码。我们可以看到，在构造模型开始之前，日志会打出一个字符串`Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}`来提示构造的模型的类型和名称。我们先来测试一下下述代码：
``` python
print("----------------------开始监视代码----------------------")
logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
print("----------------------我的分割线1----------------------")
model = build_model(config)
print("----------------------我的分割线2----------------------")
model.cuda()
print("----------------------我的分割线3----------------------")
logger.info(str(model))
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
[2021-12-31 11:40:55 swin_tiny_patch4_window7_224](main.py 137): INFO Creating model:swin/swin_tiny_patch4_window7_224
----------------------我的分割线1----------------------
----------------------我的分割线2----------------------
----------------------我的分割线3----------------------
[2021-12-31 11:40:55 swin_tiny_patch4_window7_224](main.py 143): INFO SwinTransformer(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
    (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
  )
  (pos_drop): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0): BasicLayer(
      dim=96, input_resolution=(56, 56), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=96, input_resolution=(56, 56), num_heads=3, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=96, window_size=(7, 7), num_heads=3
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=96, out_features=384, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=384, out_features=96, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=96, input_resolution=(56, 56), num_heads=3, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=96, window_size=(7, 7), num_heads=3
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=96, out_features=384, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=384, out_features=96, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(56, 56), dim=96
        (reduction): Linear(in_features=384, out_features=192, bias=False)
        (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      )
    )
    (1): BasicLayer(
      dim=192, input_resolution=(28, 28), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=192, input_resolution=(28, 28), num_heads=6, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(7, 7), num_heads=6
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=192, input_resolution=(28, 28), num_heads=6, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(7, 7), num_heads=6
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(28, 28), dim=192
        (reduction): Linear(in_features=768, out_features=384, bias=False)
        (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
    )
    (2): BasicLayer(
      dim=384, input_resolution=(14, 14), depth=6
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(7, 7), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(7, 7), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (2): SwinTransformerBlock(
          dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(7, 7), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (3): SwinTransformerBlock(
          dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(7, 7), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (4): SwinTransformerBlock(
          dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(7, 7), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (5): SwinTransformerBlock(
          dim=384, input_resolution=(14, 14), num_heads=12, window_size=7, shift_size=3, mlp_ratio=4.0
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(7, 7), num_heads=12
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(14, 14), dim=384
        (reduction): Linear(in_features=1536, out_features=768, bias=False)
        (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
      )
    )
    (3): BasicLayer(
      dim=768, input_resolution=(7, 7), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=768, input_resolution=(7, 7), num_heads=24, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=768, window_size=(7, 7), num_heads=24
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=768, input_resolution=(7, 7), num_heads=24, window_size=7, shift_size=0, mlp_ratio=4.0
          (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=768, window_size=(7, 7), num_heads=24
            (qkv): Linear(in_features=768, out_features=2304, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=768, out_features=768, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath()
          (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (act): GELU()
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
    )
  )
  (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  (avgpool): AdaptiveAvgPool1d(output_size=1)
  (head): Linear(in_features=768, out_features=1000, bias=True)
)
----------------------结束监视代码----------------------
```
由此就清楚了：在本次测试中，我们构造的Swin Transformer模型为`swin/swin_tiny_patch4_window7_224`模型，它的详细结构如上面print出来的样子。

构造模型的核心代码是`model = build_model(config)`这句话。我们进入到`build_model()`函数里面（这个函数在`/Swin-Transformer/models/build.py`脚本里），详细地看看这个函数的代码。下面就是`/Swin-Transformer/models/build.py`脚本的全部代码：
``` python
from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "swin":
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
    elif model_type == "swin_mlp":
        model = SwinMLP(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
            in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
            depths=config.MODEL.SWIN_MLP.DEPTHS,
            num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
            window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN_MLP.APE,
            patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
```
经测试，`model_type == "swin"`的值为`True`。因此，本次测试中，将要构建的Swin Transformer模型是`SwinTransformer`类的实例。也就是说，本次测试中，接下来会运行下面的这行代码：
``` python
model = SwinTransformer(
    img_size=config.DATA.IMG_SIZE,
    patch_size=config.MODEL.SWIN.PATCH_SIZE,
    in_chans=config.MODEL.SWIN.IN_CHANS,
    num_classes=config.MODEL.NUM_CLASSES,
    embed_dim=config.MODEL.SWIN.EMBED_DIM,
    depths=config.MODEL.SWIN.DEPTHS,
    num_heads=config.MODEL.SWIN.NUM_HEADS,
    window_size=config.MODEL.SWIN.WINDOW_SIZE,
    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
    qk_scale=config.MODEL.SWIN.QK_SCALE,
    drop_rate=config.MODEL.DROP_RATE,
    drop_path_rate=config.MODEL.DROP_PATH_RATE,
    ape=config.MODEL.SWIN.APE,
    patch_norm=config.MODEL.SWIN.PATCH_NORM,
    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
)
```
我们进入到`SwinTransformer`类的代码里，来看一下这个`SwinTransformer`类的完整代码。`SwinTransformer`类的完整代码位于`/Swin-Transformer/models/swin_transformer.py`脚本里，如下：
``` python
class SwinTransformer(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2 ** self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops
```
这段代码就是Swin Transformer模型的代码。我们先来看模型构造部分的代码（也就是`def __init__(...)`函数的代码）。首先初始化了一些参数：
``` python
self.num_classes = num_classes
self.num_layers = len(depths)
self.embed_dim = embed_dim
self.ape = ape
self.patch_norm = patch_norm
self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
self.mlp_ratio = mlp_ratio
```
这些参数的具体含义，参见论文作者给出的注释。