# Swin Transformer代码阅读笔记 

在这个笔记中，我将逐行分析Swin Transformer的训练代码，梳理代码结构，逐个搞懂其中用到的函数和标准库的用法。


## 训练入口

Swin Transformer的训练入口在main.py文件里。我在训练时，执行的命令是：

``` bash 
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --cfg configs/swin_tiny_patch4_window7_224.yaml --data-path /XXX-XXX/XXX_XXX_data/imagenet/origin --batch-size 32
```
（注：我的batch size设置成了32。这样做的目的是为了避免显存不够的错误。如果采用Github上原始的128[（参见Swin-T）](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md#training-from-scratch)会发生显存溢出的错误。这里采用单GPU训练。）

main.py文件由以下几个函数组成：
``` python
def parse_option():

def main(config):

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):

@torch.no_grad()
def validate(config, data_loader, model):

@torch.no_grad()
def throughput(data_loader, model, logger):

if __name__ == '__main__':
```



训练入口处的代码如下（关于训练入口行的详细含义，参见[这里](https://stackoverflow.com/questions/419163/what-does-if-name-main-do)）：
``` python
if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
```

接下来逐行分析训练入口处的代码。


在Swin Transformer开始训练时，首先执行：`_, config = parse_option()` 这个代码调用了函数`parse_option()`，该函数的代码如下：
``` python
def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config
```
返回值`args`的类型是`<class 'argparse.Namespace'>`，返回值`config`的类型是`<class 'yacs.config.CfgNode'>`，在训练入口处，把返回值`config`传给变量`config`（也就是代码中的`_, config = parse_option()`），之后将会使用`config`这个变量来调用传进来的参数。

接下来两行和混合精度训练有关：
``` python
if config.AMP_OPT_LEVEL != "O0":
    assert amp is not None, "amp not installed!"
```
`config.AMP_OPT_LEVEL`参数默认是01，不过平时我一般不会用到混合精度训练。

下面是检查环境变量中是否有`'RANK'`和`'WORLD_SIZE'`这两个key（注意：`os.environ`是一个字典，调用的时候使用`os.environ.get('RANK')`来取出环境变量所对应的值）：
``` python
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
else:
    rank = -1
    world_size = -1
```
实际测试运行`print(os.environ.get('RANK')) print(os.environ.get('WORLD_SIZE'))`，可以得到结果分别为0和1


接下来的代码和分布式训练有关（详细内容请参阅[PyTorch分布式训练文档](https://pytorch.org/docs/stable/distributed.html)）：
``` python
torch.cuda.set_device(config.LOCAL_RANK)
torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.distributed.barrier()  #同步所有进程，参见https://pytorch.org/docs/stable/distributed.html#torch.distributed.barrier
```
运行`print(config.LOCAL_RANK) print(world_size) print(rank)`之后可以得到0，1，0。（关于PyTorch分布式训练的内容，以后再写专门的文章来补全。此处按下不表。）

下面的代码设置了随机种子：
``` python
seed = config.SEED + dist.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True
```
实际结果是：
config.SEED:  0
dist.get_rank():  0
seed:  0
注意：为什么对于不同的进程要设置不同的随机种子？[这篇文章](https://zhuanlan.zhihu.com/p/368916180)给出的解释是：
```
假如你的model中用到了随机数种子来保证可复现性, 那么此时我们不能再用固定的常数作为seed, 否则会导致DDP中的所有进程都拥有一样的seed, 进而生成同态性的数据。
```
因此，对不同的进程，需要设置不同的随机种子。


下面的代码是对学习率做的线性缩放：
``` python
# linear scale the learning rate according to total batch size, may not be optimal
linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
```
其中dist.get_world_size()是1。
注意：config.TRAIN.BASE_LR等参数是在config.py文件里设置的。这些参数的设置代码是一些固定的用法，只需要记住就行了（参考[这里](https://cloud.tencent.com/developer/article/1583189)或[这里](https://blog.csdn.net/gefeng1209/article/details/90668882)）。


下面的部分还是和学习率的改变以及参数的设定相关：
``` python
if config.TRAIN.ACCUMULATION_STEPS > 1:
    linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
config.defrost()
config.TRAIN.BASE_LR = linear_scaled_lr
config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
config.TRAIN.MIN_LR = linear_scaled_min_lr
config.freeze()
```
注意：第一次运行时，`config.TRAIN.ACCUMULATION_STEPS: 0`。所以没有执行`if`里面的部分。`config.defrost()`和`config.freeze()`分别是解冻和冻结参数。修改完参数以后，把参数冻结。


下面这些代码是为了编辑日志并保存参数：
``` python
os.makedirs(config.OUTPUT, exist_ok=True)
logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

# print config
logger.info(config.dump())
```
在这里，我要补充说明一下我对这段代码的研究思路。当遇到不懂的代码时，我采用的方法是“庖丁解牛”，就是说，利用一些技术手段，对代码进行拆解和细化，逐个逐个地弄懂自己不懂的部分（个人感觉，我采用的方法，在哲学上也是有根据可循的。可以参考现代科学的先驱弗朗西斯·培根(Francis Bacon)写的《新工具》一书，以及其他一些科学哲学或科学史的书）。具体来说，我采取的办法是，把自己不懂的部分都`print`出来，看一看如果把我不懂的部分`print`出来，会发生什么。之后再查找一些相关资料，就很容易弄懂了。比如，我们来试运行一下下面的这段代码：
``` python
os.makedirs(config.OUTPUT, exist_ok=True)
logger = create_logger(
    output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}"
)

print("-------------------我的输出----------------------")
print("dist.get_rank(): ", dist.get_rank())
print("-------------------我的输出----------------------")
print("config的类型: ", type(config))
print("-------------------我的输出----------------------")
print("config.dump()的类型: ", type(config.dump()))
print("-------------------我的输出----------------------")
print("config.dump()在这里：\n", config.dump())
print("-------------------我的输出----------------------")
print("config.OUTPUT: ", config.OUTPUT)
print("-------------------我的输出----------------------")
print("type(logger): ", type(logger))
print("-------------------我的输出----------------------")
print("logger: ", logger)
print("-------------------我的输出----------------------")
exit()
```
得到的结果是：
```
-------------------我的输出----------------------
dist.get_rank():  0
-------------------我的输出----------------------
config的类型:  <class 'yacs.config.CfgNode'>
-------------------我的输出----------------------
config.dump()的类型:  <class 'str'>
-------------------我的输出----------------------
config.dump()在这里：
 AMP_OPT_LEVEL: O1
AUG:
  AUTO_AUGMENT: rand-m9-mstd0.5-inc1
  COLOR_JITTER: 0.4
  CUTMIX: 1.0
  CUTMIX_MINMAX: null
  MIXUP: 0.8
  MIXUP_MODE: batch
  MIXUP_PROB: 1.0
  MIXUP_SWITCH_PROB: 0.5
  RECOUNT: 1
  REMODE: pixel
  REPROB: 0.25
BASE:
- ''
DATA:
  BATCH_SIZE: 32
  CACHE_MODE: part
  DATASET: imagenet
  DATA_PATH: /XXX-XXX/XXX_XXX_data/imagenet/origin
  IMG_SIZE: 224
  INTERPOLATION: bicubic
  NUM_WORKERS: 8
  PIN_MEMORY: true
  ZIP_MODE: false
EVAL_MODE: false
LOCAL_RANK: 0
MODEL:
  DROP_PATH_RATE: 0.2
  DROP_RATE: 0.0
  LABEL_SMOOTHING: 0.1
  NAME: swin_tiny_patch4_window7_224
  NUM_CLASSES: 1000
  RESUME: ''
  SWIN:
    APE: false
    DEPTHS:
    - 2
    - 2
    - 6
    - 2
    EMBED_DIM: 96
    IN_CHANS: 3
    MLP_RATIO: 4.0
    NUM_HEADS:
    - 3
    - 6
    - 12
    - 24
    PATCH_NORM: true
    PATCH_SIZE: 4
    QKV_BIAS: true
    QK_SCALE: null
    WINDOW_SIZE: 7
  SWIN_MLP:
    APE: false
    DEPTHS:
    - 2
    - 2
    - 6
    - 2
    EMBED_DIM: 96
    IN_CHANS: 3
    MLP_RATIO: 4.0
    NUM_HEADS:
    - 3
    - 6
    - 12
    - 24
    PATCH_NORM: true
    PATCH_SIZE: 4
    WINDOW_SIZE: 7
  TYPE: swin
OUTPUT: output/swin_tiny_patch4_window7_224/4_epochs
PRINT_FREQ: 10
SAVE_FREQ: 1
SEED: 0
TAG: 4_epochs
TEST:
  CROP: true
THROUGHPUT_MODE: false
TRAIN:
  ACCUMULATION_STEPS: 0
  AUTO_RESUME: true
  BASE_LR: 3.125e-05
  CLIP_GRAD: 5.0
  EPOCHS: 4
  LR_SCHEDULER:
    DECAY_EPOCHS: 30
    DECAY_RATE: 0.1
    NAME: cosine
  MIN_LR: 3.125e-07
  OPTIMIZER:
    BETAS:
    - 0.9
    - 0.999
    EPS: 1.0e-08
    MOMENTUM: 0.9
    NAME: adamw
  START_EPOCH: 0
  USE_CHECKPOINT: false
  WARMUP_EPOCHS: 20
  WARMUP_LR: 3.125e-08
  WEIGHT_DECAY: 0.05

-------------------我的输出----------------------
config.OUTPUT:  output/swin_tiny_patch4_window7_224/4_epochs
-------------------我的输出----------------------
type(logger):  <class 'logging.Logger'>
-------------------我的输出----------------------
logger:  <Logger swin_tiny_patch4_window7_224 (DEBUG)>
-------------------我的输出----------------------
```
由上述演示，可以得出如下的结论：
`config`的类型是:  `<class 'yacs.config.CfgNode'>`；
`config.dump()`的类型是:  `<class 'str'>`；这就是说，`config.dump()`是把`config`这样一个树形结构转换成了一个字符串。
`logger`的类型是：`<class 'logging.Logger'>`；
`logger`的内容是：`<Logger swin_tiny_patch4_window7_224 (DEBUG)>`。

我们再来试运行一下下面的这段代码：
``` python
print("-------------------开始监视代码----------------------")

if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

print("-------------------我的输出----------------------")
print("config.OUTPUT: ", config.OUTPUT)
print("-------------------我的输出----------------------")
print("path: ", path)
print("-------------------我的输出----------------------")
logger.info("注意！注意！我现在正在对代码进行测试。哈哈哈哈。")
print("-------------------我的输出----------------------")
exit()
```
得到了如下的输出：
```
-------------------开始监视代码----------------------
[2021-10-28 12:22:58 swin_tiny_patch4_window7_224](main.py 526): INFO Full config saved to output/swin_tiny_patch4_window7_224/4_epochs/config.json
-------------------我的输出----------------------
config.OUTPUT:  output/swin_tiny_patch4_window7_224/4_epochs
-------------------我的输出----------------------
path:  output/swin_tiny_patch4_window7_224/4_epochs/config.json
-------------------我的输出----------------------
[2021-10-28 12:22:58 swin_tiny_patch4_window7_224](main.py 533): INFO 注意！注意！我现在正在对代码进行测试。哈哈哈哈。
-------------------我的输出----------------------
```
由此可知：
`config.OUTPUT`参数的值为`output/swin_tiny_patch4_window7_224/4_epochs`；
`path`的值为`output/swin_tiny_patch4_window7_224/4_epochs/config.json`；
`logger.info(str)`可以把字符串`str`按照日志的格式输出到屏幕上。
这里还需要再解释一下`config.OUTPUT`的由来。我们可以看到，`config.OUTPUT`参数的值为`output/swin_tiny_patch4_window7_224/4_epochs`，这个值是由三部分组成的（以`/`为分界线）。在`/Swin-Transformer/config.py`文件里，我们可以看到如下的代码：
``` python
def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()
```
注意这一行：
``` python
# output folder
config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
```
至此，`config.OUTPUT`的三个部分的由来就一目了然了。在我的`/Swin-Transformer/config.py`文件里，`_C.OUTPUT = ''`，因此`config.OUTPUT`的第一部分会在`/Swin-Transformer/main.py`的`parse_option()`函数里得到设定（一般就是设为默认值`"output"`）。因为`_C.MODEL.NAME = "swin_tiny_patch4_window7_224"`，所以`config.OUTPUT`的第二部分自然就是`swin_tiny_patch4_window7_224`，因为`_C.TAG = "4_epochs"`，所以`config.OUTPUT`的第三部分自然就是`4_epochs`。至此，我们终于算是彻底弄清楚了该如何设置训练完的模型所要保存的路径。即：只需设定好`_C.OUTPUT`,`_C.MODEL.NAME`和`_C.TAG`这三个参数，就确定好了训练完的模型所要保存的路径。

关于日志的输出，我们可以再来测试一下。试运行如下代码：
``` python
print("-------------------开始监视代码----------------------")

os.makedirs(config.OUTPUT, exist_ok=True)
logger = create_logger(
    output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}"
)

print("-------------------我的输出----------------------")
print(type(logger))
print("-------------------我的输出----------------------")
print(logger)
print("-------------------我的输出----------------------")
logger.info("我现在想要测试一下日志输出的用法")
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
-------------------我的输出----------------------
<class 'logging.Logger'>
-------------------我的输出----------------------
<Logger swin_tiny_patch4_window7_224 (DEBUG)>
-------------------我的输出----------------------
[2021-10-28 15:25:36 swin_tiny_patch4_window7_224](main.py 510): INFO 我现在想要测试一下日志输出的用法
-------------------结束监视代码----------------------
```
由此可见，`logger.info(str)`函数接受一个字符串`str`作为输入，并将这个字符串`str`作为日志打印到终端上。

接下来我们进入主函数`main(config)`的部分

## 主函数main(config)
主函数只有这一句话：
``` python
main(config)
```
我们首先再来回顾一下主函数接受的输入config。运行如下的几行代码：
``` python
print("-------------------我的输出----------------------")
print(type(config))
print("-------------------我的输出----------------------")
print(config)
print("-------------------我的输出----------------------")
exit()
```
得到了结果：
``` 
-------------------我的输出----------------------
<class 'yacs.config.CfgNode'>
-------------------我的输出----------------------
AMP_OPT_LEVEL: O1
AUG:
  AUTO_AUGMENT: rand-m9-mstd0.5-inc1
  COLOR_JITTER: 0.4
  CUTMIX: 1.0
  CUTMIX_MINMAX: None
  MIXUP: 0.8
  MIXUP_MODE: batch
  MIXUP_PROB: 1.0
  MIXUP_SWITCH_PROB: 0.5
  RECOUNT: 1
  REMODE: pixel
  REPROB: 0.25
BASE: ['']
DATA:
  BATCH_SIZE: 32
  CACHE_MODE: part
  DATASET: imagenet
  DATA_PATH: /XXX-XXX/XXX_XXX_data/imagenet/origin
  IMG_SIZE: 224
  INTERPOLATION: bicubic
  NUM_WORKERS: 8
  PIN_MEMORY: True
  ZIP_MODE: False
EVAL_MODE: False
LOCAL_RANK: 0
MODEL:
  DROP_PATH_RATE: 0.2
  DROP_RATE: 0.0
  LABEL_SMOOTHING: 0.1
  NAME: swin_tiny_patch4_window7_224
  NUM_CLASSES: 1000
  RESUME: 
  SWIN:
    APE: False
    DEPTHS: [2, 2, 6, 2]
    EMBED_DIM: 96
    IN_CHANS: 3
    MLP_RATIO: 4.0
    NUM_HEADS: [3, 6, 12, 24]
    PATCH_NORM: True
    PATCH_SIZE: 4
    QKV_BIAS: True
    QK_SCALE: None
    WINDOW_SIZE: 7
  SWIN_MLP:
    APE: False
    DEPTHS: [2, 2, 6, 2]
    EMBED_DIM: 96
    IN_CHANS: 3
    MLP_RATIO: 4.0
    NUM_HEADS: [3, 6, 12, 24]
    PATCH_NORM: True
    PATCH_SIZE: 4
    WINDOW_SIZE: 7
  TYPE: swin
OUTPUT: output/swin_tiny_patch4_window7_224/4_epochs
PRINT_FREQ: 10
SAVE_FREQ: 1
SEED: 0
TAG: 4_epochs
TEST:
  CROP: True
THROUGHPUT_MODE: False
TRAIN:
  ACCUMULATION_STEPS: 0
  AUTO_RESUME: True
  BASE_LR: 3.125e-05
  CLIP_GRAD: 5.0
  EPOCHS: 4
  LR_SCHEDULER:
    DECAY_EPOCHS: 30
    DECAY_RATE: 0.1
    NAME: cosine
  MIN_LR: 3.125e-07
  OPTIMIZER:
    BETAS: (0.9, 0.999)
    EPS: 1e-08
    MOMENTUM: 0.9
    NAME: adamw
  START_EPOCH: 0
  USE_CHECKPOINT: False
  WARMUP_EPOCHS: 20
  WARMUP_LR: 3.125e-08
  WEIGHT_DECAY: 0.05
-------------------我的输出----------------------
```
通过如上的演示，我们可以对config这个对象有更加深刻的认识。config本质上就是一个把各种参数打包后形成的一个树形结构。真正要调用各个具体的参数的时候呢，使用点语法即可。如下所示：
``` python
print(config.DATA.BATCH_SIZE == 32)
print(config.MODEL.NAME == "swin_tiny_patch4_window7_224")
print(config.MODEL.SWIN_MLP.DEPTHS == [2, 2, 6, 2])
```
结果都是True。

我们终于来到了主函数的部分。主函数`main()`的完整代码如下：
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

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=config.AMP_OPT_LEVEL
        )
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, logger
        )
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
        )
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
        )
        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
        ):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                max_accuracy,
                optimizer,
                lr_scheduler,
                logger,
            )

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
        )
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
```
接下来我们继续一行一行地来分析主函数的运行过程，我们的目标是：弄懂其中每一行的含义和用法。

首先，主函数执行了这样的一行代码：
``` python
(
    dataset_train,
    dataset_val,
    data_loader_train,
    data_loader_val,
    mixup_fn,
) = build_loader(config)
```
这行代码的核心是，把`config`对象传给了`build_loader()`函数，返回了五个之后训练要用到的对象。`build_loader()`函数位于`/Swin-Transformer/data/build.py`里，是在main.py的开头由`from data import build_loader`导入的。`build_loader()`函数的代码如下：
``` python
def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
```
build_loader(config)函数的前三行如下：
``` python
config.defrost()
dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
config.freeze()
```
第一行和第三行分别是对参数对象config执行解冻和冻结操作。config对象解冻以后，才可以对config里的参数进行修改。修改完以后，再把config对象冻结，防止对里面的参数进行修改。最核心的第二行，把`True`和`config`对象传给`build_dataset()`函数，返回值赋给了`dataset_train`和`config.MODEL.NUM_CLASSES`。我们来看`build_dataset()`函数的详细代码：
``` python
def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes
```
`build_dataset()`函数的第一行又调用了`build_transform()`函数。`build_transform()`函数的代码如下：
``` python
def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
```
我们下面来详细分析`build_transform()`函数每一行的含义。

`build_transform()`函数的第一行代码`resize_im = config.DATA.IMG_SIZE > 32`会返回一个bool值给`resize_im`，在我上面的设置中，`config.DATA.IMG_SIZE: 224`，所以，`resize_im`的值应为`True`

接下来的代码是`if is_train:`包裹的代码：
``` python
if is_train:
    # this should always dispatch to transforms_imagenet_train
    transform = create_transform(
        input_size=config.DATA.IMG_SIZE,
        is_training=True,
        color_jitter=config.AUG.COLOR_JITTER
        if config.AUG.COLOR_JITTER > 0
        else None,
        auto_augment=config.AUG.AUTO_AUGMENT
        if config.AUG.AUTO_AUGMENT != "none"
        else None,
        re_prob=config.AUG.REPROB,
        re_mode=config.AUG.REMODE,
        re_count=config.AUG.RECOUNT,
        interpolation=config.DATA.INTERPOLATION,
    )
    if not resize_im:
        # replace RandomResizedCropAndInterpolation with
        # RandomCrop
        transform.transforms[0] = transforms.RandomCrop(
            config.DATA.IMG_SIZE, padding=4
        )
    return transform
```
因为`is_train==True  resize_im==True`，所以会进入到这段代码里来执行。首先执行：
``` python
transform = create_transform(
    input_size=config.DATA.IMG_SIZE,
    is_training=True,
    color_jitter=config.AUG.COLOR_JITTER
    if config.AUG.COLOR_JITTER > 0
    else None,
    auto_augment=config.AUG.AUTO_AUGMENT
    if config.AUG.AUTO_AUGMENT != "none"
    else None,
    re_prob=config.AUG.REPROB,
    re_mode=config.AUG.REMODE,
    re_count=config.AUG.RECOUNT,
    interpolation=config.DATA.INTERPOLATION,
)
```
这段代码使用`config`对象里的各种参数，来初始化一个`transform`对象。`create_transform()`这个函数是由`/Swin-Transformer/data/build.py`文件里的`from timm.data import create_transform`语句导入的。试运行一下下面的代码：
``` python
print("-------------------开始监视代码----------------------")
transform = create_transform(
    input_size=config.DATA.IMG_SIZE,
    is_training=True,
    color_jitter=config.AUG.COLOR_JITTER
    if config.AUG.COLOR_JITTER > 0
    else None,
    auto_augment=config.AUG.AUTO_AUGMENT
    if config.AUG.AUTO_AUGMENT != "none"
    else None,
    re_prob=config.AUG.REPROB,
    re_mode=config.AUG.REMODE,
    re_count=config.AUG.RECOUNT,
    interpolation=config.DATA.INTERPOLATION,
)
print("-------------------我的输出----------------------")
print(type(transform))
print("-------------------我的输出----------------------")
print(transform)
print("-------------------结束监视代码----------------------")
exit()
```
得到的输出为：
```
-------------------开始监视代码----------------------
-------------------我的输出----------------------
<class 'torchvision.transforms.transforms.Compose'>
-------------------我的输出----------------------
Compose(
    RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC)
    RandomXXXtalFlip(p=0.5)
    <timm.data.auto_augment.RandAugment object at 0x7f909edfa450>
    ToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
    <timm.data.random_erasing.RandomErasing object at 0x7f909e772490>
)
-------------------结束监视代码----------------------
```
在这之后，由于`resize_im==True`，所以下面这段代码：
``` python
if not resize_im:
    # replace RandomResizedCropAndInterpolation with
    # RandomCrop
    transform.transforms[0] = transforms.RandomCrop(
        config.DATA.IMG_SIZE, padding=4
    )
```
不会被执行，因此直接返回`transform`对象，`build_transform(is_train, config)`函数执行完毕。如果是在测试阶段，也就是不需要训练的阶段，`is_train==False`，因此在执行`build_transform(is_train, config)`函数的时候，会执行`resize_im = config.DATA.IMG_SIZE > 32`这一行，然后就直接进入`t = []`及以后的部分，不会执行第一个return。

至此我们可以回到`build_dataset(is_train, config)`函数的执行过程了。

在`build_dataset(is_train, config)`函数中，第一行代码`transform = build_transform(is_train, config)`已经执行完毕了，并且返回了一个`transform`对象赋值给`transform`变量。我们再次试运行一下下面的代码：
``` python
def build_dataset(is_train, config):
    print("-------------------开始监视代码----------------------")
    transform = build_transform(is_train, config)
    print("-------------------我的分割线1----------------------")
    print(type(transform))
    print("-------------------我的分割线2----------------------")
    print(transform)
    print("-------------------结束监视代码----------------------")
    exit()
```
得到的结果是：
```
-------------------开始监视代码----------------------
-------------------我的分割线1----------------------
<class 'torchvision.transforms.transforms.Compose'>
-------------------我的分割线2----------------------
Compose(
    RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC)
    RandomXXXtalFlip(p=0.5)
    <timm.data.auto_augment.RandAugment object at 0x7f64c429e790>
    ToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
    <timm.data.random_erasing.RandomErasing object at 0x7f6537abee10>
)
-------------------结束监视代码----------------------
```

由于表达式`config.DATA.DATASET == "imagenet"`的结果为`True`，所以接下来将会执行如下的这段代码：
``` python
prefix = "train" if is_train else "val"
if config.DATA.ZIP_MODE:
    ann_file = prefix + "_map.txt"
    prefix = prefix + ".zip@/"
    dataset = CachedImageFolder(
        config.DATA.DATA_PATH,
        ann_file,
        prefix,
        transform,
        cache_mode=config.DATA.CACHE_MODE if is_train else "part",
    )
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
nb_classes = 1000
```
`prefix`这个英语单词自身是`前缀`的意思，因此在训练阶段（也就是`is_train==True`时），`prefix`这个变量会被设为字符串`"train"`；如果是在非训练阶段（也就是`is_train==False`时），`prefix`这个变量会被设为字符串`"val"`，也就是用于表示`验证`阶段。
接下来试运行一下下面的这段代码：
``` python
prefix = "train" if is_train else "val"
print("-------------------开始监视代码----------------------")
print(config.DATA.ZIP_MODE)
print("-------------------结束监视代码----------------------")
exit()
if config.DATA.ZIP_MODE:
```
结果为：
```
-------------------开始监视代码----------------------
False
-------------------结束监视代码----------------------
```
`config.DATA.ZIP_MODE`这个变量为`False`，说明本次训练使用的数据不是压缩文件，而是存储在文件夹中的数据（参见`/Swin-Transformer/config.py`中下面的这个代码片段：）
```
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
```
因此本次执行时，会执行这三行：
``` python
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
nb_classes = 1000
```
试运行下面的这段代码：
``` python
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    print("-------------------开始监视代码----------------------")
    print(root)
    print("-------------------结束监视代码----------------------")
    exit()
    dataset = datasets.ImageFolder(root, transform=transform)
nb_classes = 1000
```
结果为：
```
-------------------开始监视代码----------------------
/XXX-XXX/XXX_XXX_data/imagenet/origin/train
-------------------结束监视代码----------------------
```
这个`root`路径就是我要使用的训练数据存放的地方。如果实际地进入路径`/XXX-XXX/XXX_XXX_data/imagenet/origin/train`看一下的话，这个路径下面的内容会是这个样子的：
```
gen_train.sh  n02033041  n02124075  n02797295  n03425413  n03929660  n04429376
n01440764     n02037110  n02125311  n02799071  n03443371  n03929855  n04435653
n01443537     n02051845  n02127052  n02802426  n03444034  n03930313  n04442312
n01484850     n02056570  n02128385  n02804414  n03445777  n03930630  n04443257
n01491361     n02058221  n02128757  n02804610  n03445924  n03933933  n04447861
n01494475     n02066245  n02128925  n02807133  n03447447  n03935335  n04456115
n01496331     n02071294  n02129165  n02808304  n03447721  n03937543  n04458633
n01498041     n02074367  n02129604  n02808440  n03450230  n03938244  n04461696
n01514668     n02077923  n02130308  n02814533  n03452741  n03942813  n04462240
n01514859     n02085620  n02132136  n02814860  n03457902  n03944341  n04465501
n01518878     n02085782  n02133161  n02815834  n03459775  n03947888  n04467665
n01530575     n02085936  n02134084  n02817516  n03461385  n03950228  n04476259
n01531178     n02086079  n02134418  n02823428  n03467068  n03954731  n04479046
n01532829     n02086240  n02137549  n02823750  n03476684  n03956157  n04482393
n01534433     n02086646  n02138441  n02825657  n03476991  n03958227  n04483307
n01537544     n02086910  n02165105  n02834397  n03478589  n03961711  n04485082
n01558993     n02087046  n02165456  n02835271  n03481172  n03967562  n04486054
n01560419     n02087394  n02167151  n02837789  n03482405  n03970156  n04487081
n01580077     n02088094  n02168699  n02840245  n03483316  n03976467  n04487394
n01582220     n02088238  n02169497  n02841315  n03485407  n03976657  n04493381
n01592084     n02088364  n02172182  n02843684  n03485794  n03977966  n04501370
n01601694     n02088466  n02174001  n02859443  n03492542  n03980874  n04505470
n01608432     n02088632  n02177972  n02860847  n03494278  n03982430  n04507155
n01614925     n02089078  n02190166  n02865351  n03495258  n03983396  n04509417
n01616318     n02089867  n02206856  n02869837  n03496892  n03991062  n04515003
n01622779     n02089973  n02219486  n02870880  n03498962  n03992509  n04517823
n01629819     n02090379  n02226429  n02871525  n03527444  n03995372  n04522168
n01630670     n02090622  n02229544  n02877765  n03529860  n03998194  n04523525
n01631663     n02090721  n02231487  n02879718  n03530642  n04004767  n04525038
n01632458     n02091032  n02233338  n02883205  n03532672  n04005630  n04525305
n01632777     n02091134  n02236044  n02892201  n03534580  n04008634  n04532106
n01641577     n02091244  n02256656  n02892767  n03535780  n04009552  n04532670
n01644373     n02091467  n02259212  n02894605  n03538406  n04019541  n04536866
n01644900     n02091635  n02264363  n02895154  n03544143  n04023962  n04540053
n01664065     n02091831  n02268443  n02906734  n03584254  n04026417  n04542943
n01665541     n02092002  n02268853  n02909870  n03584829  n04033901  n04548280
n01667114     n02092339  n02276258  n02910353  n03590841  n04033995  n04548362
n01667778     n02093256  n02277742  n02916936  n03594734  n04037443  n04550184
n01669191     n02093428  n02279972  n02917067  n03594945  n04039381  n04552348
n01675722     n02093647  n02280649  n02927161  n03595614  n04040759  n04553703
n01677366     n02093754  n02281406  n02930766  n03598930  n04041544  n04554684
n01682714     n02093859  n02281787  n02939185  n03599486  n04044716  n04557648
n01685808     n02093991  n02317335  n02948072  n03602883  n04049303  n04560804
n01687978     n02094114  n02319095  n02950826  n03617480  n04065272  n04562935
n01688243     n02094258  n02321529  n02951358  n03623198  n04067472  n04579145
n01689811     n02094433  n02325366  n02951585  n03627232  n04069434  n04579432
n01692333     n02095314  n02326432  n02963159  n03630383  n04070727  n04584207
n01693334     n02095570  n02328150  n02965783  n03633091  n04074963  n04589890
n01694178     n02095889  n02342885  n02966193  n03637318  n04081281  n04590129
n01695060     n02096051  n02346627  n02966687  n03642806  n04086273  n04591157
n01697457     n02096177  n02356798  n02971356  n03649909  n04090263  n04591713
n01698640     n02096294  n02361337  n02974003  n03657121  n04099969  n04592741
n01704323     n02096437  n02363005  n02977058  n03658185  n04111531  n04596742
n01728572     n02096585  n02364673  n02978881  n03661043  n04116512  n04597913
n01728920     n02097047  n02389026  n02979186  n03662601  n04118538  n04599235
n01729322     n02097130  n02391049  n02980441  n03666591  n04118776  n04604644
n01729977     n02097209  n02395406  n02981792  n03670208  n04120489  n04606251
n01734418     n02097298  n02396427  n02988304  n03673027  n04125021  n04612504
n01735189     n02097474  n02397096  n02992211  n03676483  n04127249  n04613696
n01737021     n02097658  n02398521  n02992529  n03680355  n04131690  n06359193
n01739381     n02098105  n02403003  n02999410  n03690938  n04133789  n06596364
n01740131     n02098286  n02408429  n03000134  n03691459  n04136333  n06785654
n01742172     n02098413  n02410509  n03000247  n03692522  n04141076  n06794110
n01744401     n02099267  n02412080  n03000684  n03697007  n04141327  n06874185
n01748264     n02099429  n02415577  n03014705  n03706229  n04141975  n07248320
n01749939     n02099601  n02417914  n03016953  n03709823  n04146614  n07565083
n01751748     n02099712  n02422106  n03017168  n03710193  n04147183  n07579787
n01753488     n02099849  n02422699  n03018349  n03710637  n04149813  n07583066
n01755581     n02100236  n02423022  n03026506  n03710721  n04152593  n07584110
n01756291     n02100583  n02437312  n03028079  n03717622  n04153751  n07590611
n01768244     n02100735  n02437616  n03032252  n03720891  n04154565  n07613480
n01770081     n02100877  n02441942  n03041632  n03721384  n04162706  n07614500
n01770393     n02101006  n02442845  n03042490  n03724870  n04179913  n07615774
n01773157     n02101388  n02443114  n03045698  n03729826  n04192698  n07684084
n01773549     n02101556  n02443484  n03047690  n03733131  n04200800  n07693725
n01773797     n02102040  n02444819  n03062245  n03733281  n04201297  n07695742
n01774384     n02102177  n02445715  n03063599  n03733805  n04204238  n07697313
n01774750     n02102318  n02447366  n03063689  n03742115  n04204347  n07697537
n01775062     n02102480  n02454379  n03065424  n03743016  n04208210  n07711569
n01776313     n02102973  n02457408  n03075370  n03759954  n04209133  n07714571
n01784675     n02104029  n02480495  n03085013  n03761084  n04209239  n07714990
n01795545     n02104365  n02480855  n03089624  n03763968  n04228054  n07715103
n01796340     n02105056  n02481823  n03095699  n03764736  n04229816  n07716358
n01797886     n02105162  n02483362  n03100240  n03769881  n04235860  n07716906
n01798484     n02105251  n02483708  n03109150  n03770439  n04238763  n07717410
n01806143     n02105412  n02484975  n03110669  n03770679  n04239074  n07717556
n01806567     n02105505  n02486261  n03124043  n03773504  n04243546  n07718472
n01807496     n02105641  n02486410  n03124170  n03775071  n04251144  n07718747
n01817953     n02105855  n02487347  n03125729  n03775546  n04252077  n07720875
n01818515     n02106030  n02488291  n03126707  n03776460  n04252225  n07730033
n01819313     n02106166  n02488702  n03127747  n03777568  n04254120  n07734744
n01820546     n02106382  n02489166  n03127925  n03777754  n04254680  n07742313
n01824575     n02106550  n02490219  n03131574  n03781244  n04254777  n07745940
n01828970     n02106662  n02492035  n03133878  n03782006  n04258138  n07747607
n01829413     n02107142  n02492660  n03134739  n03785016  n04259630  n07749582
n01833805     n02107312  n02493509  n03141823  n03786901  n04263257  n07753113
n01843065     n02107574  n02493793  n03146219  n03787032  n04264628  n07753275
n01843383     n02107683  n02494079  n03160309  n03788195  n04265275  n07753592
n01847000     n02107908  n02497673  n03179701  n03788365  n04266014  n07754684
n01855032     n02108000  n02500267  n03180011  n03791053  n04270147  n07760859
n01855672     n02108089  n02504013  n03187595  n03792782  n04273569  n07768694
n01860187     n02108422  n02504458  n03188531  n03792972  n04275548  n07802026
n01871265     n02108551  n02509815  n03196217  n03793489  n04277352  n07831146
n01872401     n02108915  n02510455  n03197337  n03794056  n04285008  n07836838
n01873310     n02109047  n02514041  n03201208  n03796401  n04286575  n07860988
n01877812     n02109525  n02526121  n03207743  n03803284  n04296562  n07871810
n01882714     n02109961  n02536864  n03207941  n03804744  n04310018  n07873807
n01883070     n02110063  n02606052  n03208938  n03814639  n04311004  n07875152
n01910747     n02110185  n02607072  n03216828  n03814906  n04311174  n07880968
n01914609     n02110341  n02640242  n03218198  n03825788  n04317175  n07892512
n01917289     n02110627  n02641379  n03220513  n03832673  n04325704  n07920052
n01924916     n02110806  n02643566  n03223299  n03837869  n04326547  n07930864
n01930112     n02110958  n02655020  n03240683  n03838899  n04328186  n07932039
n01943899     n02111129  n02666196  n03249569  n03840681  n04330267  n09193705
n01944390     n02111277  n02667093  n03250847  n03841143  n04332243  n09229709
n01945685     n02111500  n02669723  n03255030  n03843555  n04335435  n09246464
n01950731     n02111889  n02672831  n03259280  n03854065  n04336792  n09256479
n01955084     n02112018  n02676566  n03271574  n03857828  n04344873  n09288635
n01968897     n02112137  n02687172  n03272010  n03866082  n04346328  n09332890
n01978287     n02112350  n02690373  n03272562  n03868242  n04347754  n09399592
n01978455     n02112706  n02692877  n03290653  n03868863  n04350905  n09421951
n01980166     n02113023  n02699494  n03291819  n03871628  n04355338  n09428293
n01981276     n02113186  n02701002  n03297495  n03873416  n04355933  n09468604
n01983481     n02113624  n02704792  n03314780  n03874293  n04356056  n09472597
n01984695     n02113712  n02708093  n03325584  n03874599  n04357314  n09835506
n01985128     n02113799  n02727426  n03337140  n03876231  n04366367  n10148035
n01986214     n02113978  n02730930  n03344393  n03877472  n04367480  n10565667
n01990800     n02114367  n02747177  n03345487  n03877845  n04370456  n11879895
n02002556     n02114548  n02749479  n03347037  n03884397  n04371430  n11939491
n02002724     n02114712  n02769748  n03355925  n03887697  n04371774  n12057211
n02006656     n02114855  n02776631  n03372029  n03888257  n04372370  n12144580
n02007558     n02115641  n02777292  n03376595  n03888605  n04376876  n12267677
n02009229     n02115913  n02782093  n03379051  n03891251  n04380533  n12620546
n02009912     n02116738  n02783161  n03384352  n03891332  n04389033  n12768682
n02011460     n02117135  n02786058  n03388043  n03895866  n04392985  n12985857
n02012849     n02119022  n02787622  n03388183  n03899768  n04398044  n12998815
n02013706     n02119789  n02788148  n03388549  n03902125  n04399382  n13037406
n02017213     n02120079  n02790996  n03393912  n03903868  n04404412  n13040303
n02018207     n02120505  n02791124  n03394916  n03908618  n04409515  n13044778
n02018795     n02123045  n02791270  n03400231  n03908714  n04417672  n13052670
n02025239     n02123159  n02793495  n03404251  n03916031  n04418357  n13054560
n02027492     n02123394  n02794156  n03417042  n03920288  n04423845  n13133613
n02028035     n02123597  n02795169  n03424325  n03924679  n04428191  n15075141
```
除去第一个shell脚本以外，这里的每一个条目都是一个文件夹，每个文件夹代表一个物体的类别（虽然文件夹的名字并不是该类物体的名称。这一点对于人类的理解确实有一些负面影响，但不影响网络的训练）。这里的这个ImageNet数据集是ImageNet-1k数据集。我们再任意地进入一个文件夹，比如进入`n02033041`，里面的内容是这样的：
```
n02033041_10047.JPEG  n02033041_1933.JPEG  n02033041_3747.JPEG  n02033041_5942.JPEG
n02033041_10053.JPEG  n02033041_1938.JPEG  n02033041_3749.JPEG  n02033041_5963.JPEG
n02033041_1006.JPEG   n02033041_1939.JPEG  n02033041_3750.JPEG  n02033041_5967.JPEG
n02033041_100.JPEG    n02033041_1945.JPEG  n02033041_3753.JPEG  n02033041_5968.JPEG
n02033041_10105.JPEG  n02033041_1947.JPEG  n02033041_3757.JPEG  n02033041_5977.JPEG
n02033041_1010.JPEG   n02033041_1958.JPEG  n02033041_3781.JPEG  n02033041_5981.JPEG
n02033041_10120.JPEG  n02033041_195.JPEG   n02033041_3786.JPEG  n02033041_5992.JPEG
n02033041_10122.JPEG  n02033041_1960.JPEG  n02033041_3790.JPEG  n02033041_5994.JPEG
n02033041_1012.JPEG   n02033041_1965.JPEG  n02033041_3800.JPEG  n02033041_5.JPEG
n02033041_10153.JPEG  n02033041_1980.JPEG  n02033041_3805.JPEG  n02033041_6017.JPEG
n02033041_10169.JPEG  n02033041_198.JPEG   n02033041_3806.JPEG  n02033041_6036.JPEG
n02033041_10170.JPEG  n02033041_1991.JPEG  n02033041_382.JPEG   n02033041_603.JPEG
n02033041_1017.JPEG   n02033041_1995.JPEG  n02033041_3833.JPEG  n02033041_6041.JPEG
n02033041_10181.JPEG  n02033041_2001.JPEG  n02033041_3834.JPEG  n02033041_6042.JPEG
n02033041_1018.JPEG   n02033041_201.JPEG   n02033041_3839.JPEG  n02033041_6046.JPEG
n02033041_10213.JPEG  n02033041_2025.JPEG  n02033041_3847.JPEG  n02033041_6062.JPEG
n02033041_10228.JPEG  n02033041_2031.JPEG  n02033041_384.JPEG   n02033041_6065.JPEG
n02033041_1022.JPEG   n02033041_2032.JPEG  n02033041_3851.JPEG  n02033041_6069.JPEG
n02033041_10234.JPEG  n02033041_2034.JPEG  n02033041_385.JPEG   n02033041_6105.JPEG
n02033041_1026.JPEG   n02033041_2036.JPEG  n02033041_3867.JPEG  n02033041_6116.JPEG
n02033041_1028.JPEG   n02033041_2042.JPEG  n02033041_3885.JPEG  n02033041_6117.JPEG
n02033041_1031.JPEG   n02033041_2052.JPEG  n02033041_3887.JPEG  n02033041_6121.JPEG
n02033041_10321.JPEG  n02033041_2055.JPEG  n02033041_3889.JPEG  n02033041_6124.JPEG
n02033041_1036.JPEG   n02033041_2065.JPEG  n02033041_3895.JPEG  n02033041_6131.JPEG
n02033041_10463.JPEG  n02033041_2066.JPEG  n02033041_3913.JPEG  n02033041_6132.JPEG
n02033041_10484.JPEG  n02033041_2069.JPEG  n02033041_3916.JPEG  n02033041_6133.JPEG
n02033041_10490.JPEG  n02033041_2072.JPEG  n02033041_3918.JPEG  n02033041_6143.JPEG
n02033041_10516.JPEG  n02033041_2074.JPEG  n02033041_3919.JPEG  n02033041_6148.JPEG
n02033041_10527.JPEG  n02033041_2077.JPEG  n02033041_3922.JPEG  n02033041_6150.JPEG
n02033041_10536.JPEG  n02033041_2079.JPEG  n02033041_3924.JPEG  n02033041_6151.JPEG
n02033041_10553.JPEG  n02033041_2082.JPEG  n02033041_3926.JPEG  n02033041_616.JPEG
n02033041_10620.JPEG  n02033041_2089.JPEG  n02033041_3932.JPEG  n02033041_617.JPEG
n02033041_10657.JPEG  n02033041_2092.JPEG  n02033041_3935.JPEG  n02033041_6187.JPEG
n02033041_10660.JPEG  n02033041_2098.JPEG  n02033041_3940.JPEG  n02033041_6192.JPEG
n02033041_10695.JPEG  n02033041_2101.JPEG  n02033041_3952.JPEG  n02033041_619.JPEG
n02033041_106.JPEG    n02033041_2119.JPEG  n02033041_3954.JPEG  n02033041_6214.JPEG
n02033041_1073.JPEG   n02033041_2120.JPEG  n02033041_3961.JPEG  n02033041_6223.JPEG
n02033041_1074.JPEG   n02033041_2131.JPEG  n02033041_3970.JPEG  n02033041_6231.JPEG
n02033041_10750.JPEG  n02033041_2135.JPEG  n02033041_3978.JPEG  n02033041_6234.JPEG
n02033041_1078.JPEG   n02033041_2139.JPEG  n02033041_3989.JPEG  n02033041_6236.JPEG
n02033041_1088.JPEG   n02033041_2142.JPEG  n02033041_398.JPEG   n02033041_6256.JPEG
n02033041_1089.JPEG   n02033041_2143.JPEG  n02033041_3992.JPEG  n02033041_6263.JPEG
n02033041_1091.JPEG   n02033041_214.JPEG   n02033041_4001.JPEG  n02033041_626.JPEG
n02033041_10997.JPEG  n02033041_2160.JPEG  n02033041_4007.JPEG  n02033041_6282.JPEG
n02033041_10.JPEG     n02033041_2164.JPEG  n02033041_4023.JPEG  n02033041_6287.JPEG
n02033041_11014.JPEG  n02033041_2167.JPEG  n02033041_4026.JPEG  n02033041_6299.JPEG
n02033041_11015.JPEG  n02033041_2171.JPEG  n02033041_4032.JPEG  n02033041_6317.JPEG
n02033041_11018.JPEG  n02033041_2184.JPEG  n02033041_4037.JPEG  n02033041_6318.JPEG
n02033041_1105.JPEG   n02033041_2188.JPEG  n02033041_403.JPEG   n02033041_6320.JPEG
n02033041_1107.JPEG   n02033041_2199.JPEG  n02033041_4043.JPEG  n02033041_6321.JPEG
n02033041_11107.JPEG  n02033041_21.JPEG    n02033041_404.JPEG   n02033041_6343.JPEG
n02033041_1111.JPEG   n02033041_2217.JPEG  n02033041_4050.JPEG  n02033041_6349.JPEG
n02033041_1113.JPEG   n02033041_2218.JPEG  n02033041_4094.JPEG  n02033041_634.JPEG
n02033041_11197.JPEG  n02033041_2219.JPEG  n02033041_4095.JPEG  n02033041_6354.JPEG
n02033041_11213.JPEG  n02033041_2222.JPEG  n02033041_4102.JPEG  n02033041_6356.JPEG
n02033041_11258.JPEG  n02033041_2224.JPEG  n02033041_4111.JPEG  n02033041_6362.JPEG
n02033041_1126.JPEG   n02033041_2229.JPEG  n02033041_4122.JPEG  n02033041_6370.JPEG
n02033041_11284.JPEG  n02033041_2232.JPEG  n02033041_4123.JPEG  n02033041_6375.JPEG
n02033041_11353.JPEG  n02033041_2233.JPEG  n02033041_412.JPEG   n02033041_6384.JPEG
n02033041_11370.JPEG  n02033041_2235.JPEG  n02033041_413.JPEG   n02033041_6391.JPEG
n02033041_11378.JPEG  n02033041_2240.JPEG  n02033041_4140.JPEG  n02033041_6399.JPEG
n02033041_1138.JPEG   n02033041_2242.JPEG  n02033041_4155.JPEG  n02033041_640.JPEG
n02033041_11403.JPEG  n02033041_224.JPEG   n02033041_415.JPEG   n02033041_641.JPEG
n02033041_11415.JPEG  n02033041_2250.JPEG  n02033041_4161.JPEG  n02033041_6420.JPEG
n02033041_1141.JPEG   n02033041_2271.JPEG  n02033041_4165.JPEG  n02033041_6424.JPEG
n02033041_1142.JPEG   n02033041_2275.JPEG  n02033041_4179.JPEG  n02033041_6425.JPEG
n02033041_1145.JPEG   n02033041_2295.JPEG  n02033041_4180.JPEG  n02033041_6427.JPEG
n02033041_11467.JPEG  n02033041_2312.JPEG  n02033041_4184.JPEG  n02033041_6441.JPEG
n02033041_11477.JPEG  n02033041_2320.JPEG  n02033041_4194.JPEG  n02033041_6446.JPEG
n02033041_11478.JPEG  n02033041_2328.JPEG  n02033041_4225.JPEG  n02033041_6463.JPEG
n02033041_11492.JPEG  n02033041_2331.JPEG  n02033041_4229.JPEG  n02033041_6465.JPEG
n02033041_1155.JPEG   n02033041_2333.JPEG  n02033041_4234.JPEG  n02033041_646.JPEG
n02033041_11580.JPEG  n02033041_2341.JPEG  n02033041_423.JPEG   n02033041_6470.JPEG
n02033041_115.JPEG    n02033041_234.JPEG   n02033041_4243.JPEG  n02033041_647.JPEG
n02033041_11660.JPEG  n02033041_2356.JPEG  n02033041_4261.JPEG  n02033041_648.JPEG
n02033041_11665.JPEG  n02033041_2359.JPEG  n02033041_4273.JPEG  n02033041_6493.JPEG
n02033041_116.JPEG    n02033041_235.JPEG   n02033041_428.JPEG   n02033041_649.JPEG
n02033041_11729.JPEG  n02033041_2364.JPEG  n02033041_429.JPEG   n02033041_6517.JPEG
n02033041_1172.JPEG   n02033041_2367.JPEG  n02033041_4307.JPEG  n02033041_6519.JPEG
n02033041_11742.JPEG  n02033041_2376.JPEG  n02033041_4319.JPEG  n02033041_6523.JPEG
n02033041_11806.JPEG  n02033041_2378.JPEG  n02033041_4337.JPEG  n02033041_6533.JPEG
n02033041_11822.JPEG  n02033041_2387.JPEG  n02033041_435.JPEG   n02033041_6563.JPEG
n02033041_11837.JPEG  n02033041_2391.JPEG  n02033041_436.JPEG   n02033041_6564.JPEG
n02033041_11868.JPEG  n02033041_2394.JPEG  n02033041_437.JPEG   n02033041_6565.JPEG
n02033041_11872.JPEG  n02033041_2400.JPEG  n02033041_4383.JPEG  n02033041_6582.JPEG
n02033041_11887.JPEG  n02033041_2402.JPEG  n02033041_438.JPEG   n02033041_660.JPEG
n02033041_1189.JPEG   n02033041_2406.JPEG  n02033041_4394.JPEG  n02033041_6614.JPEG
n02033041_1191.JPEG   n02033041_2410.JPEG  n02033041_4407.JPEG  n02033041_6616.JPEG
n02033041_1193.JPEG   n02033041_2412.JPEG  n02033041_4412.JPEG  n02033041_6620.JPEG
n02033041_1197.JPEG   n02033041_2413.JPEG  n02033041_4448.JPEG  n02033041_6633.JPEG
n02033041_12030.JPEG  n02033041_2418.JPEG  n02033041_4455.JPEG  n02033041_6642.JPEG
n02033041_12048.JPEG  n02033041_2430.JPEG  n02033041_4458.JPEG  n02033041_6645.JPEG
n02033041_12073.JPEG  n02033041_2435.JPEG  n02033041_4461.JPEG  n02033041_6655.JPEG
n02033041_1207.JPEG   n02033041_2437.JPEG  n02033041_4463.JPEG  n02033041_6657.JPEG
n02033041_12113.JPEG  n02033041_2439.JPEG  n02033041_449.JPEG   n02033041_6665.JPEG
n02033041_12135.JPEG  n02033041_2440.JPEG  n02033041_4500.JPEG  n02033041_6675.JPEG
n02033041_1213.JPEG   n02033041_2442.JPEG  n02033041_4513.JPEG  n02033041_6676.JPEG
n02033041_12165.JPEG  n02033041_2445.JPEG  n02033041_4514.JPEG  n02033041_667.JPEG
n02033041_12248.JPEG  n02033041_2446.JPEG  n02033041_451.JPEG   n02033041_6686.JPEG
n02033041_12328.JPEG  n02033041_2451.JPEG  n02033041_4527.JPEG  n02033041_668.JPEG
n02033041_1233.JPEG   n02033041_2452.JPEG  n02033041_4531.JPEG  n02033041_6694.JPEG
n02033041_1234.JPEG   n02033041_2456.JPEG  n02033041_4541.JPEG  n02033041_6704.JPEG
n02033041_12354.JPEG  n02033041_2457.JPEG  n02033041_456.JPEG   n02033041_6711.JPEG
n02033041_12390.JPEG  n02033041_2464.JPEG  n02033041_4591.JPEG  n02033041_671.JPEG
n02033041_12426.JPEG  n02033041_2470.JPEG  n02033041_4618.JPEG  n02033041_6720.JPEG
n02033041_12463.JPEG  n02033041_2472.JPEG  n02033041_4634.JPEG  n02033041_6728.JPEG
n02033041_1248.JPEG   n02033041_2474.JPEG  n02033041_4642.JPEG  n02033041_6733.JPEG
n02033041_12509.JPEG  n02033041_2480.JPEG  n02033041_4644.JPEG  n02033041_6739.JPEG
n02033041_12553.JPEG  n02033041_2489.JPEG  n02033041_4653.JPEG  n02033041_6743.JPEG
n02033041_1257.JPEG   n02033041_249.JPEG   n02033041_4675.JPEG  n02033041_6763.JPEG
n02033041_1258.JPEG   n02033041_2502.JPEG  n02033041_46.JPEG    n02033041_6785.JPEG
n02033041_1259.JPEG   n02033041_2514.JPEG  n02033041_4715.JPEG  n02033041_6786.JPEG
n02033041_12673.JPEG  n02033041_2516.JPEG  n02033041_4719.JPEG  n02033041_6796.JPEG
n02033041_12674.JPEG  n02033041_2521.JPEG  n02033041_4738.JPEG  n02033041_6797.JPEG
n02033041_1267.JPEG   n02033041_2525.JPEG  n02033041_4749.JPEG  n02033041_6841.JPEG
n02033041_12710.JPEG  n02033041_2528.JPEG  n02033041_4751.JPEG  n02033041_6855.JPEG
n02033041_12737.JPEG  n02033041_2530.JPEG  n02033041_4760.JPEG  n02033041_6856.JPEG
n02033041_1275.JPEG   n02033041_2534.JPEG  n02033041_4764.JPEG  n02033041_6864.JPEG
n02033041_12794.JPEG  n02033041_2536.JPEG  n02033041_4780.JPEG  n02033041_6867.JPEG
n02033041_127.JPEG    n02033041_2538.JPEG  n02033041_4789.JPEG  n02033041_6869.JPEG
n02033041_12878.JPEG  n02033041_2540.JPEG  n02033041_4791.JPEG  n02033041_6892.JPEG
n02033041_12880.JPEG  n02033041_2543.JPEG  n02033041_4798.JPEG  n02033041_689.JPEG
n02033041_128.JPEG    n02033041_2552.JPEG  n02033041_4800.JPEG  n02033041_6902.JPEG
n02033041_12932.JPEG  n02033041_2564.JPEG  n02033041_4804.JPEG  n02033041_6906.JPEG
n02033041_12961.JPEG  n02033041_2565.JPEG  n02033041_4827.JPEG  n02033041_6909.JPEG
n02033041_1296.JPEG   n02033041_2567.JPEG  n02033041_4838.JPEG  n02033041_6919.JPEG
n02033041_12974.JPEG  n02033041_2575.JPEG  n02033041_4839.JPEG  n02033041_6922.JPEG
n02033041_1298.JPEG   n02033041_257.JPEG   n02033041_4841.JPEG  n02033041_6924.JPEG
n02033041_12998.JPEG  n02033041_2596.JPEG  n02033041_4853.JPEG  n02033041_692.JPEG
n02033041_13009.JPEG  n02033041_2600.JPEG  n02033041_4863.JPEG  n02033041_6934.JPEG
n02033041_1301.JPEG   n02033041_260.JPEG   n02033041_4868.JPEG  n02033041_693.JPEG
n02033041_13059.JPEG  n02033041_2615.JPEG  n02033041_486.JPEG   n02033041_6946.JPEG
n02033041_13074.JPEG  n02033041_2616.JPEG  n02033041_487.JPEG   n02033041_6971.JPEG
n02033041_130.JPEG    n02033041_2624.JPEG  n02033041_488.JPEG   n02033041_6974.JPEG
n02033041_1316.JPEG   n02033041_2627.JPEG  n02033041_4890.JPEG  n02033041_6985.JPEG
n02033041_13187.JPEG  n02033041_2629.JPEG  n02033041_4901.JPEG  n02033041_6990.JPEG
n02033041_13209.JPEG  n02033041_2633.JPEG  n02033041_4913.JPEG  n02033041_6994.JPEG
n02033041_1323.JPEG   n02033041_2641.JPEG  n02033041_4914.JPEG  n02033041_6996.JPEG
n02033041_13245.JPEG  n02033041_2648.JPEG  n02033041_4918.JPEG  n02033041_699.JPEG
n02033041_13304.JPEG  n02033041_2658.JPEG  n02033041_4925.JPEG  n02033041_69.JPEG
n02033041_13310.JPEG  n02033041_2659.JPEG  n02033041_492.JPEG   n02033041_7018.JPEG
n02033041_13315.JPEG  n02033041_2670.JPEG  n02033041_4932.JPEG  n02033041_7019.JPEG
n02033041_13327.JPEG  n02033041_2671.JPEG  n02033041_4938.JPEG  n02033041_7021.JPEG
n02033041_13331.JPEG  n02033041_2672.JPEG  n02033041_4939.JPEG  n02033041_7038.JPEG
n02033041_13448.JPEG  n02033041_2673.JPEG  n02033041_493.JPEG   n02033041_7044.JPEG
n02033041_1345.JPEG   n02033041_2675.JPEG  n02033041_4942.JPEG  n02033041_704.JPEG
n02033041_1352.JPEG   n02033041_2679.JPEG  n02033041_4943.JPEG  n02033041_7085.JPEG
n02033041_1356.JPEG   n02033041_267.JPEG   n02033041_4948.JPEG  n02033041_708.JPEG
n02033041_1357.JPEG   n02033041_2680.JPEG  n02033041_495.JPEG   n02033041_7090.JPEG
n02033041_13585.JPEG  n02033041_2682.JPEG  n02033041_4964.JPEG  n02033041_7093.JPEG
n02033041_1359.JPEG   n02033041_2689.JPEG  n02033041_4967.JPEG  n02033041_7096.JPEG
n02033041_13608.JPEG  n02033041_2690.JPEG  n02033041_496.JPEG   n02033041_7117.JPEG
n02033041_13626.JPEG  n02033041_2697.JPEG  n02033041_4970.JPEG  n02033041_7129.JPEG
n02033041_13629.JPEG  n02033041_2703.JPEG  n02033041_4975.JPEG  n02033041_7138.JPEG
n02033041_13673.JPEG  n02033041_2704.JPEG  n02033041_4978.JPEG  n02033041_713.JPEG
n02033041_136.JPEG    n02033041_2710.JPEG  n02033041_4989.JPEG  n02033041_7148.JPEG
n02033041_13755.JPEG  n02033041_2714.JPEG  n02033041_4991.JPEG  n02033041_7182.JPEG
n02033041_13860.JPEG  n02033041_2722.JPEG  n02033041_4993.JPEG  n02033041_7193.JPEG
n02033041_1389.JPEG   n02033041_2733.JPEG  n02033041_4997.JPEG  n02033041_7196.JPEG
n02033041_13993.JPEG  n02033041_2737.JPEG  n02033041_5000.JPEG  n02033041_7200.JPEG
n02033041_14074.JPEG  n02033041_2739.JPEG  n02033041_5003.JPEG  n02033041_7205.JPEG
n02033041_14083.JPEG  n02033041_2742.JPEG  n02033041_5007.JPEG  n02033041_720.JPEG
n02033041_1409.JPEG   n02033041_2745.JPEG  n02033041_5009.JPEG  n02033041_7219.JPEG
n02033041_140.JPEG    n02033041_2748.JPEG  n02033041_501.JPEG   n02033041_7220.JPEG
n02033041_14144.JPEG  n02033041_2755.JPEG  n02033041_5020.JPEG  n02033041_7229.JPEG
n02033041_14158.JPEG  n02033041_2763.JPEG  n02033041_5026.JPEG  n02033041_7230.JPEG
n02033041_14236.JPEG  n02033041_2770.JPEG  n02033041_5031.JPEG  n02033041_7236.JPEG
n02033041_14299.JPEG  n02033041_2772.JPEG  n02033041_503.JPEG   n02033041_7244.JPEG
n02033041_14315.JPEG  n02033041_2774.JPEG  n02033041_5040.JPEG  n02033041_7246.JPEG
n02033041_14316.JPEG  n02033041_2775.JPEG  n02033041_5045.JPEG  n02033041_7250.JPEG
n02033041_14335.JPEG  n02033041_2779.JPEG  n02033041_5047.JPEG  n02033041_7252.JPEG
n02033041_14366.JPEG  n02033041_2796.JPEG  n02033041_5050.JPEG  n02033041_7259.JPEG
n02033041_14374.JPEG  n02033041_2805.JPEG  n02033041_5059.JPEG  n02033041_7266.JPEG
n02033041_14377.JPEG  n02033041_2806.JPEG  n02033041_505.JPEG   n02033041_7269.JPEG
n02033041_14395.JPEG  n02033041_2826.JPEG  n02033041_5066.JPEG  n02033041_7277.JPEG
n02033041_14447.JPEG  n02033041_2835.JPEG  n02033041_5069.JPEG  n02033041_7281.JPEG
n02033041_1449.JPEG   n02033041_2836.JPEG  n02033041_5073.JPEG  n02033041_7285.JPEG
n02033041_1450.JPEG   n02033041_284.JPEG   n02033041_5079.JPEG  n02033041_7290.JPEG
n02033041_1453.JPEG   n02033041_2876.JPEG  n02033041_5086.JPEG  n02033041_7294.JPEG
n02033041_14656.JPEG  n02033041_2888.JPEG  n02033041_5088.JPEG  n02033041_7299.JPEG
n02033041_14681.JPEG  n02033041_2889.JPEG  n02033041_508.JPEG   n02033041_7300.JPEG
n02033041_1468.JPEG   n02033041_2891.JPEG  n02033041_5091.JPEG  n02033041_7316.JPEG
n02033041_14714.JPEG  n02033041_2901.JPEG  n02033041_5098.JPEG  n02033041_7319.JPEG
n02033041_14727.JPEG  n02033041_2906.JPEG  n02033041_5100.JPEG  n02033041_7331.JPEG
n02033041_14777.JPEG  n02033041_2909.JPEG  n02033041_5108.JPEG  n02033041_7335.JPEG
n02033041_14823.JPEG  n02033041_2925.JPEG  n02033041_511.JPEG   n02033041_7339.JPEG
n02033041_1482.JPEG   n02033041_2926.JPEG  n02033041_5130.JPEG  n02033041_7344.JPEG
n02033041_1489.JPEG   n02033041_2928.JPEG  n02033041_513.JPEG   n02033041_734.JPEG
n02033041_14924.JPEG  n02033041_2933.JPEG  n02033041_514.JPEG   n02033041_7352.JPEG
n02033041_1492.JPEG   n02033041_2934.JPEG  n02033041_5161.JPEG  n02033041_7355.JPEG
n02033041_1495.JPEG   n02033041_2945.JPEG  n02033041_5163.JPEG  n02033041_7378.JPEG
n02033041_14985.JPEG  n02033041_2946.JPEG  n02033041_5175.JPEG  n02033041_7409.JPEG
n02033041_1501.JPEG   n02033041_2948.JPEG  n02033041_5176.JPEG  n02033041_7415.JPEG
n02033041_15026.JPEG  n02033041_2950.JPEG  n02033041_5183.JPEG  n02033041_7416.JPEG
n02033041_15036.JPEG  n02033041_2952.JPEG  n02033041_5197.JPEG  n02033041_7420.JPEG
n02033041_1504.JPEG   n02033041_2955.JPEG  n02033041_51.JPEG    n02033041_7430.JPEG
n02033041_1505.JPEG   n02033041_2958.JPEG  n02033041_5208.JPEG  n02033041_7432.JPEG
n02033041_15072.JPEG  n02033041_2960.JPEG  n02033041_5209.JPEG  n02033041_743.JPEG
n02033041_15073.JPEG  n02033041_2968.JPEG  n02033041_5233.JPEG  n02033041_7444.JPEG
n02033041_15091.JPEG  n02033041_2969.JPEG  n02033041_523.JPEG   n02033041_7469.JPEG
n02033041_1509.JPEG   n02033041_296.JPEG   n02033041_5240.JPEG  n02033041_7476.JPEG
n02033041_15110.JPEG  n02033041_2971.JPEG  n02033041_5249.JPEG  n02033041_7477.JPEG
n02033041_15162.JPEG  n02033041_297.JPEG   n02033041_5250.JPEG  n02033041_7488.JPEG
n02033041_1516.JPEG   n02033041_2982.JPEG  n02033041_5257.JPEG  n02033041_7502.JPEG
n02033041_1519.JPEG   n02033041_2990.JPEG  n02033041_525.JPEG   n02033041_7507.JPEG
n02033041_15260.JPEG  n02033041_2991.JPEG  n02033041_5260.JPEG  n02033041_7563.JPEG
n02033041_1526.JPEG   n02033041_2995.JPEG  n02033041_5263.JPEG  n02033041_7579.JPEG
n02033041_15297.JPEG  n02033041_2996.JPEG  n02033041_5264.JPEG  n02033041_7585.JPEG
n02033041_1530.JPEG   n02033041_2999.JPEG  n02033041_5279.JPEG  n02033041_7624.JPEG
n02033041_1535.JPEG   n02033041_3006.JPEG  n02033041_5280.JPEG  n02033041_7625.JPEG
n02033041_1536.JPEG   n02033041_3009.JPEG  n02033041_5281.JPEG  n02033041_7681.JPEG
n02033041_15377.JPEG  n02033041_3017.JPEG  n02033041_5290.JPEG  n02033041_770.JPEG
n02033041_1541.JPEG   n02033041_3022.JPEG  n02033041_5300.JPEG  n02033041_7729.JPEG
n02033041_1543.JPEG   n02033041_3050.JPEG  n02033041_5303.JPEG  n02033041_7734.JPEG
n02033041_15444.JPEG  n02033041_3059.JPEG  n02033041_5304.JPEG  n02033041_7740.JPEG
n02033041_15453.JPEG  n02033041_3073.JPEG  n02033041_5306.JPEG  n02033041_7755.JPEG
n02033041_1549.JPEG   n02033041_3078.JPEG  n02033041_530.JPEG   n02033041_7756.JPEG
n02033041_15505.JPEG  n02033041_3083.JPEG  n02033041_5313.JPEG  n02033041_7761.JPEG
n02033041_1550.JPEG   n02033041_3087.JPEG  n02033041_5319.JPEG  n02033041_7783.JPEG
n02033041_1553.JPEG   n02033041_3098.JPEG  n02033041_5331.JPEG  n02033041_7796.JPEG
n02033041_15629.JPEG  n02033041_3109.JPEG  n02033041_5332.JPEG  n02033041_7801.JPEG
n02033041_15634.JPEG  n02033041_3114.JPEG  n02033041_5334.JPEG  n02033041_781.JPEG
n02033041_15638.JPEG  n02033041_3119.JPEG  n02033041_5335.JPEG  n02033041_7825.JPEG
n02033041_1563.JPEG   n02033041_3121.JPEG  n02033041_5340.JPEG  n02033041_7852.JPEG
n02033041_15656.JPEG  n02033041_3136.JPEG  n02033041_534.JPEG   n02033041_7900.JPEG
n02033041_1566.JPEG   n02033041_3140.JPEG  n02033041_5363.JPEG  n02033041_790.JPEG
n02033041_156.JPEG    n02033041_3150.JPEG  n02033041_5364.JPEG  n02033041_7951.JPEG
n02033041_1571.JPEG   n02033041_3151.JPEG  n02033041_5368.JPEG  n02033041_7957.JPEG
n02033041_15732.JPEG  n02033041_3163.JPEG  n02033041_5373.JPEG  n02033041_795.JPEG
n02033041_1573.JPEG   n02033041_3165.JPEG  n02033041_5375.JPEG  n02033041_7992.JPEG
n02033041_15743.JPEG  n02033041_3168.JPEG  n02033041_5386.JPEG  n02033041_7.JPEG
n02033041_15745.JPEG  n02033041_3177.JPEG  n02033041_5387.JPEG  n02033041_809.JPEG
n02033041_1583.JPEG   n02033041_3179.JPEG  n02033041_5388.JPEG  n02033041_8112.JPEG
n02033041_15878.JPEG  n02033041_317.JPEG   n02033041_538.JPEG   n02033041_812.JPEG
n02033041_15902.JPEG  n02033041_3183.JPEG  n02033041_5390.JPEG  n02033041_8148.JPEG
n02033041_15910.JPEG  n02033041_3185.JPEG  n02033041_5399.JPEG  n02033041_820.JPEG
n02033041_15918.JPEG  n02033041_318.JPEG   n02033041_539.JPEG   n02033041_8251.JPEG
n02033041_15963.JPEG  n02033041_3192.JPEG  n02033041_5401.JPEG  n02033041_8272.JPEG
n02033041_16018.JPEG  n02033041_3196.JPEG  n02033041_5403.JPEG  n02033041_8285.JPEG
n02033041_1601.JPEG   n02033041_3198.JPEG  n02033041_5409.JPEG  n02033041_8308.JPEG
n02033041_16025.JPEG  n02033041_3202.JPEG  n02033041_540.JPEG   n02033041_831.JPEG
n02033041_1602.JPEG   n02033041_3208.JPEG  n02033041_5411.JPEG  n02033041_8352.JPEG
n02033041_1604.JPEG   n02033041_3216.JPEG  n02033041_5420.JPEG  n02033041_8373.JPEG
n02033041_1607.JPEG   n02033041_3221.JPEG  n02033041_542.JPEG   n02033041_839.JPEG
n02033041_16094.JPEG  n02033041_3226.JPEG  n02033041_543.JPEG   n02033041_8434.JPEG
n02033041_16097.JPEG  n02033041_3242.JPEG  n02033041_5441.JPEG  n02033041_8437.JPEG
n02033041_1613.JPEG   n02033041_3243.JPEG  n02033041_5459.JPEG  n02033041_8500.JPEG
n02033041_16202.JPEG  n02033041_3258.JPEG  n02033041_5466.JPEG  n02033041_8518.JPEG
n02033041_1622.JPEG   n02033041_3262.JPEG  n02033041_5471.JPEG  n02033041_8544.JPEG
n02033041_16258.JPEG  n02033041_3263.JPEG  n02033041_5472.JPEG  n02033041_8581.JPEG
n02033041_1625.JPEG   n02033041_3265.JPEG  n02033041_5473.JPEG  n02033041_859.JPEG
n02033041_16264.JPEG  n02033041_3272.JPEG  n02033041_5484.JPEG  n02033041_85.JPEG
n02033041_16270.JPEG  n02033041_3274.JPEG  n02033041_5485.JPEG  n02033041_8634.JPEG
n02033041_16313.JPEG  n02033041_3276.JPEG  n02033041_5490.JPEG  n02033041_863.JPEG
n02033041_1632.JPEG   n02033041_3282.JPEG  n02033041_5495.JPEG  n02033041_8663.JPEG
n02033041_1640.JPEG   n02033041_328.JPEG   n02033041_5513.JPEG  n02033041_8677.JPEG
n02033041_1642.JPEG   n02033041_3296.JPEG  n02033041_5515.JPEG  n02033041_8684.JPEG
n02033041_16462.JPEG  n02033041_3323.JPEG  n02033041_5519.JPEG  n02033041_868.JPEG
n02033041_16467.JPEG  n02033041_3326.JPEG  n02033041_5528.JPEG  n02033041_8753.JPEG
n02033041_16475.JPEG  n02033041_3328.JPEG  n02033041_5531.JPEG  n02033041_878.JPEG
n02033041_1648.JPEG   n02033041_3329.JPEG  n02033041_5540.JPEG  n02033041_8797.JPEG
n02033041_16505.JPEG  n02033041_3331.JPEG  n02033041_5544.JPEG  n02033041_87.JPEG
n02033041_16507.JPEG  n02033041_3333.JPEG  n02033041_5547.JPEG  n02033041_8808.JPEG
n02033041_16537.JPEG  n02033041_3334.JPEG  n02033041_5550.JPEG  n02033041_880.JPEG
n02033041_165.JPEG    n02033041_333.JPEG   n02033041_5558.JPEG  n02033041_8821.JPEG
n02033041_16610.JPEG  n02033041_3340.JPEG  n02033041_5566.JPEG  n02033041_882.JPEG
n02033041_16617.JPEG  n02033041_3344.JPEG  n02033041_5576.JPEG  n02033041_883.JPEG
n02033041_1662.JPEG   n02033041_3349.JPEG  n02033041_5578.JPEG  n02033041_8866.JPEG
n02033041_16666.JPEG  n02033041_335.JPEG   n02033041_5607.JPEG  n02033041_8879.JPEG
n02033041_16669.JPEG  n02033041_3361.JPEG  n02033041_5609.JPEG  n02033041_8899.JPEG
n02033041_16676.JPEG  n02033041_3363.JPEG  n02033041_5619.JPEG  n02033041_8937.JPEG
n02033041_16718.JPEG  n02033041_336.JPEG   n02033041_5624.JPEG  n02033041_8968.JPEG
n02033041_1674.JPEG   n02033041_3372.JPEG  n02033041_563.JPEG   n02033041_9014.JPEG
n02033041_16766.JPEG  n02033041_3378.JPEG  n02033041_5642.JPEG  n02033041_9023.JPEG
n02033041_1677.JPEG   n02033041_3380.JPEG  n02033041_5645.JPEG  n02033041_9026.JPEG
n02033041_167.JPEG    n02033041_3394.JPEG  n02033041_5654.JPEG  n02033041_903.JPEG
n02033041_1683.JPEG   n02033041_3396.JPEG  n02033041_5655.JPEG  n02033041_9040.JPEG
n02033041_1684.JPEG   n02033041_339.JPEG   n02033041_5672.JPEG  n02033041_9066.JPEG
n02033041_1685.JPEG   n02033041_3406.JPEG  n02033041_5674.JPEG  n02033041_910.JPEG
n02033041_16871.JPEG  n02033041_3408.JPEG  n02033041_5679.JPEG  n02033041_911.JPEG
n02033041_1688.JPEG   n02033041_3425.JPEG  n02033041_5684.JPEG  n02033041_9125.JPEG
n02033041_168.JPEG    n02033041_3431.JPEG  n02033041_5685.JPEG  n02033041_912.JPEG
n02033041_16903.JPEG  n02033041_3435.JPEG  n02033041_5687.JPEG  n02033041_9132.JPEG
n02033041_16938.JPEG  n02033041_3442.JPEG  n02033041_56.JPEG    n02033041_9166.JPEG
n02033041_1694.JPEG   n02033041_3444.JPEG  n02033041_5707.JPEG  n02033041_9173.JPEG
n02033041_1697.JPEG   n02033041_3446.JPEG  n02033041_5724.JPEG  n02033041_9237.JPEG
n02033041_16995.JPEG  n02033041_344.JPEG   n02033041_5733.JPEG  n02033041_927.JPEG
n02033041_17099.JPEG  n02033041_3453.JPEG  n02033041_5734.JPEG  n02033041_9311.JPEG
n02033041_17119.JPEG  n02033041_3455.JPEG  n02033041_573.JPEG   n02033041_9317.JPEG
n02033041_17162.JPEG  n02033041_3457.JPEG  n02033041_5742.JPEG  n02033041_9364.JPEG
n02033041_1718.JPEG   n02033041_345.JPEG   n02033041_5743.JPEG  n02033041_9368.JPEG
n02033041_1724.JPEG   n02033041_3462.JPEG  n02033041_5744.JPEG  n02033041_937.JPEG
n02033041_1725.JPEG   n02033041_3465.JPEG  n02033041_5746.JPEG  n02033041_941.JPEG
n02033041_1736.JPEG   n02033041_3471.JPEG  n02033041_5749.JPEG  n02033041_9420.JPEG
n02033041_1745.JPEG   n02033041_3476.JPEG  n02033041_5756.JPEG  n02033041_9454.JPEG
n02033041_1753.JPEG   n02033041_3487.JPEG  n02033041_575.JPEG   n02033041_945.JPEG
n02033041_1754.JPEG   n02033041_3490.JPEG  n02033041_576.JPEG   n02033041_9465.JPEG
n02033041_1761.JPEG   n02033041_3497.JPEG  n02033041_5781.JPEG  n02033041_952.JPEG
n02033041_1763.JPEG   n02033041_349.JPEG   n02033041_5782.JPEG  n02033041_9532.JPEG
n02033041_1764.JPEG   n02033041_3501.JPEG  n02033041_5793.JPEG  n02033041_9537.JPEG
n02033041_1773.JPEG   n02033041_3505.JPEG  n02033041_5796.JPEG  n02033041_953.JPEG
n02033041_1776.JPEG   n02033041_3509.JPEG  n02033041_5797.JPEG  n02033041_9553.JPEG
n02033041_1780.JPEG   n02033041_3516.JPEG  n02033041_579.JPEG   n02033041_956.JPEG
n02033041_1782.JPEG   n02033041_3518.JPEG  n02033041_5804.JPEG  n02033041_9581.JPEG
n02033041_179.JPEG    n02033041_3519.JPEG  n02033041_5805.JPEG  n02033041_9587.JPEG
n02033041_1801.JPEG   n02033041_351.JPEG   n02033041_5813.JPEG  n02033041_95.JPEG
n02033041_1803.JPEG   n02033041_3545.JPEG  n02033041_5815.JPEG  n02033041_9600.JPEG
n02033041_1805.JPEG   n02033041_354.JPEG   n02033041_5820.JPEG  n02033041_9604.JPEG
n02033041_1824.JPEG   n02033041_3556.JPEG  n02033041_5825.JPEG  n02033041_960.JPEG
n02033041_1836.JPEG   n02033041_356.JPEG   n02033041_5827.JPEG  n02033041_961.JPEG
n02033041_1838.JPEG   n02033041_3612.JPEG  n02033041_5838.JPEG  n02033041_9633.JPEG
n02033041_1856.JPEG   n02033041_362.JPEG   n02033041_5839.JPEG  n02033041_963.JPEG
n02033041_1858.JPEG   n02033041_3638.JPEG  n02033041_5843.JPEG  n02033041_967.JPEG
n02033041_1861.JPEG   n02033041_363.JPEG   n02033041_5852.JPEG  n02033041_9701.JPEG
n02033041_1864.JPEG   n02033041_3649.JPEG  n02033041_5856.JPEG  n02033041_9728.JPEG
n02033041_1865.JPEG   n02033041_365.JPEG   n02033041_5869.JPEG  n02033041_981.JPEG
n02033041_1890.JPEG   n02033041_3665.JPEG  n02033041_5881.JPEG  n02033041_9837.JPEG
n02033041_1896.JPEG   n02033041_3670.JPEG  n02033041_5908.JPEG  n02033041_9866.JPEG
n02033041_1901.JPEG   n02033041_3674.JPEG  n02033041_5911.JPEG  n02033041_9883.JPEG
n02033041_1906.JPEG   n02033041_3679.JPEG  n02033041_5921.JPEG  n02033041_988.JPEG
n02033041_1910.JPEG   n02033041_36.JPEG    n02033041_5928.JPEG  n02033041_9904.JPEG
n02033041_1919.JPEG   n02033041_3723.JPEG  n02033041_5929.JPEG  n02033041_9910.JPEG
n02033041_1925.JPEG   n02033041_3724.JPEG  n02033041_5930.JPEG  n02033041_9924.JPEG
n02033041_1926.JPEG   n02033041_3733.JPEG  n02033041_5934.JPEG  n02033041_992.JPEG
n02033041_1927.JPEG   n02033041_3741.JPEG  n02033041_5941.JPEG  n02033041_9947.JPEG
```
任意地看几张图片，就会发现，这些图片都是水鸟的图片，各种不同的长着又长又尖嘴巴的水鸟。由此可知，我使用的ImageNet数据集的目录结构如下：
```
/XXX-XXX/XXX_XXX_data/imagenet/origin
├── train
│   ├── n02033041
│   │   ├── n02033041_10047.JPEG
│   │   ├── n02033041_1933.JPEG
│   │   └── ...
│   ├── n02124075
│   │   ├── n02124075_10012.JPEG
│   │   └── ...
│   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   └── ...
    ├── n01990800
    │   ├── ILSVRC2012_val_00000669.JPEG
    │   └── ...
    └── ...
```
可以看到，我使用的数据的目录树结构和Github上提供的Swin Transformer的作者们使用的训练数据的[目录树结构](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md#data-preparation)：
```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```
是完全相同的。所以，这大大降低了我跑通代码的难度。同时也提醒我们：要想最快速度地顺利地跑通代码，尽量使用和代码原始作者完全一样的数据组织形式（主要就是数据目录结构，或者是采用的数据列表的形式）。

接下来，我们需要引入一个更强大的解剖代码的工具：监控代码的运行时间。我采用了[这里](https://blog.csdn.net/wangshuang1631/article/details/54286551)提供的工具。我要使用这篇文章提供的方法1。

运行下述代码：
``` python
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    print("-------------------开始监视代码----------------------")
    starttime = datetime.datetime.now()
    dataset = datasets.ImageFolder(root, transform=transform)
    endtime = datetime.datetime.now()
    print(
        f"dataset = datasets.ImageFolder(root, transform=transform)这行代码的运行时间是：{(endtime - starttime).microseconds}微秒"
    )
    print(
        f"dataset = datasets.ImageFolder(root, transform=transform)这行代码的运行时间是：{(endtime - starttime).seconds}秒"
    )
    print("-------------------结束监视代码----------------------")
    exit()
nb_classes = 1000
```
结果为：
```
-------------------开始监视代码----------------------
dataset = datasets.ImageFolder(root, transform=transform)这行代码的运行时间是：440965微秒
dataset = datasets.ImageFolder(root, transform=transform)这行代码的运行时间是：6秒
-------------------结束监视代码----------------------
```
这行代码的运行时间要比之前的任何一行代码都长。我们来仔细地分析一下这行代码究竟在做什么。我们再来单独地看一下这行代码：
``` python
dataset = datasets.ImageFolder(root, transform=transform)
```
<div id="dataset_first_analysis"></div>

这行代码的核心组件是一个`datasets.ImageFolder`类（参考[这里](https://pytorch.org/vision/0.8/datasets.html#imagefolder)）。这个类是PyTorch自己已经定义好的类。关于`datasets.ImageFolder`类的中文介绍，可以参考[这里](https://blog.csdn.net/weixin_40123108/article/details/85099449)或[这里](https://blog.csdn.net/TH_NUM/article/details/80877435)

我们再次确认一下输入的root究竟是什么。试运行下述代码：
``` python
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    print("-------------------开始监视代码----------------------")
    print(root)
    print("-------------------结束监视代码----------------------")
    exit()
    dataset = datasets.ImageFolder(root, transform=transform)
nb_classes = 1000
```
结果为：
```
-------------------开始监视代码----------------------
/XXX-XXX/XXX_XXX_data/imagenet/origin/train
-------------------结束监视代码----------------------
```
一定要注意：输入到`dataset = datasets.ImageFolder(root, transform=transform)`这行代码里面的`root`，是`/XXX-XXX/XXX_XXX_data/imagenet/origin/train`。参考[这篇文章](https://blog.csdn.net/weixin_40123108/article/details/85099449)以及我上面对我自己使用的训练数据组织形式的分析可知，我的`root`目录下正好就是所有的物体类别文件夹，每个物体类别文件夹里装的都是某一个类别的物体的各种不同图片。这样的数据组织形式正好就是[这篇文章](https://blog.csdn.net/weixin_40123108/article/details/85099449)里提到的`datasets.ImageFolder()`的传入参数需要的组织形式。

再来试运行下述代码：
``` python
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    print("-------------------开始监视代码----------------------")
    print(type(transform))
    print("-------------------我的输出----------------------")
    print(transform)
    print("-------------------结束监视代码----------------------")
    exit()
    dataset = datasets.ImageFolder(root, transform=transform)
nb_classes = 1000
```
结果为：
```
-------------------开始监视代码----------------------
<class 'torchvision.transforms.transforms.Compose'>
-------------------我的输出----------------------
Compose(
    RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC)
    RandomXXXtalFlip(p=0.5)
    <timm.data.auto_augment.RandAugment object at 0x7fbbc4cf6650>
    ToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
    <timm.data.random_erasing.RandomErasing object at 0x7fbbc46d1dd0>
)
-------------------结束监视代码----------------------
```
这就是我们之前研究过的`transform`对象，这个对象被传进了`datasets.ImageFolder()`函数。至此，`datasets.ImageFolder()`函数的两个输入已经完全弄清楚了。我们终于可以来看这个函数的输出了。
<div id="datasets.ImageFolder_shili"></div>
试运行下述代码：
``` python
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
    print("-------------------开始监视代码----------------------")
    print(type(dataset))
    print("-------------------我的输出----------------------")
    print(dataset)
    print("-------------------结束监视代码----------------------")
    exit()
nb_classes = 1000
```
结果为：
```
-------------------开始监视代码----------------------
<class 'torchvision.datasets.folder.ImageFolder'>
-------------------我的输出----------------------
Dataset ImageFolder
    Number of datapoints: 1281167
    Root location: /XXX-XXX/XXX_XXX_data/imagenet/origin/train
    StandardTransform
Transform: Compose(
               RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC)
               RandomXXXtalFlip(p=0.5)
               <timm.data.auto_augment.RandAugment object at 0x7fcc9a901d50>
               ToTensor()
               Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
               <timm.data.random_erasing.RandomErasing object at 0x7fcc9a6b3390>
           )
-------------------结束监视代码----------------------
```
对于这个`dataset`对象，参考[这里](https://blog.csdn.net/TH_NUM/article/details/80877435)的介绍，进行如下的实验：
``` python
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
    print("-------------------开始监视代码----------------------")
    print("len(dataset)：", len(dataset))
    print("-------------------我的分割线1----------------------")
    print("type(dataset[0])：", type(dataset[0]))
    print("-------------------我的分割线2----------------------")
    print("len(dataset[0])：", len(dataset[0]))
    print("-------------------我的分割线3----------------------")
    print("dataset[0]：", dataset[0])
    print("-------------------我的分割线4----------------------")
    print("type(dataset[0][0])：", type(dataset[0][0]))
    print("-------------------我的分割线5----------------------")
    print("dataset[0][0].shape：", dataset[0][0].shape)
    print("-------------------我的分割线6----------------------")
    print("dataset[0][1]：", dataset[0][1])
    print("-------------------我的分割线7----------------------")
    print("type(dataset[1281166])：", type(dataset[1281166]))
    print("-------------------我的分割线8----------------------")
    print("len(dataset[1281166])：", len(dataset[1281166]))
    print("-------------------我的分割线9----------------------")
    print("dataset[1281166]：", dataset[1281166])
    print("-------------------我的分割线10----------------------")
    print("type(dataset[1281166][0])：", type(dataset[1281166][0]))
    print("-------------------我的分割线11----------------------")
    print("dataset[1281166][0].shape：", dataset[1281166][0].shape)
    print("-------------------我的分割线12----------------------")
    print("dataset[1281166][1]：", dataset[1281166][1])
    print("-------------------结束监视代码----------------------")
    exit()
nb_classes = 1000
```
结果为：
```
-------------------开始监视代码----------------------
len(dataset)： 1281167
-------------------我的分割线1----------------------
type(dataset[0])： <class 'tuple'>
-------------------我的分割线2----------------------
len(dataset[0])： 2
-------------------我的分割线3----------------------
dataset[0]： (tensor([[[ 0.0056,  0.0056,  0.0056,  ...,  0.0056,  0.0056,  0.0056],
         [ 0.0056,  0.0056,  0.0056,  ...,  0.0056,  0.0056,  0.0056],
         [ 0.0056,  0.0056,  0.0056,  ...,  0.0056,  0.0056,  0.0056],
         ...,
         [ 0.0056,  0.0056,  0.0056,  ...,  0.0056,  0.0056,  0.0056],
         [ 0.0056,  0.0056,  0.0056,  ...,  0.0056,  0.0056,  0.0056],
         [ 0.0056,  0.0056,  0.0056,  ...,  0.0056,  0.0056,  0.0056]],

        [[-0.0049, -0.0049, -0.0049,  ..., -0.0049, -0.0049, -0.0049],
         [-0.0049, -0.0049, -0.0049,  ..., -0.0049, -0.0049, -0.0049],
         [-0.0049, -0.0049, -0.0049,  ..., -0.0049, -0.0049, -0.0049],
         ...,
         [-0.0049, -0.0049, -0.0049,  ..., -0.0049, -0.0049, -0.0049],
         [-0.0049, -0.0049, -0.0049,  ..., -0.0049, -0.0049, -0.0049],
         [-0.0049, -0.0049, -0.0049,  ..., -0.0049, -0.0049, -0.0049]],

        [[ 0.0082,  0.0082,  0.0082,  ...,  0.0082,  0.0082,  0.0082],
         [ 0.0082,  0.0082,  0.0082,  ...,  0.0082,  0.0082,  0.0082],
         [ 0.0082,  0.0082,  0.0082,  ...,  0.0082,  0.0082,  0.0082],
         ...,
         [ 0.0082,  0.0082,  0.0082,  ...,  0.0082,  0.0082,  0.0082],
         [ 0.0082,  0.0082,  0.0082,  ...,  0.0082,  0.0082,  0.0082],
         [ 0.0082,  0.0082,  0.0082,  ...,  0.0082,  0.0082,  0.0082]]]), 0)
-------------------我的分割线4----------------------
type(dataset[0][0])： <class 'torch.Tensor'>
-------------------我的分割线5----------------------
dataset[0][0].shape： torch.Size([3, 224, 224])
-------------------我的分割线6----------------------
dataset[0][1]： 0
-------------------我的分割线7----------------------
type(dataset[1281166])： <class 'tuple'>
-------------------我的分割线8----------------------
len(dataset[1281166])： 2
-------------------我的分割线9----------------------
dataset[1281166]： (tensor([[[-1.7925, -1.8097,  0.0056,  ...,  0.0056,  0.0056,  0.0056],
         [-1.8097, -1.7925, -1.8268,  ...,  0.0056,  0.0056,  0.0056],
         [-1.7754, -1.7754, -1.8097,  ...,  0.0056,  0.0056,  0.0056],
         ...,
         [-1.7925, -1.7925, -1.7925,  ..., -0.5424, -0.5424, -0.5082],
         [-1.7925, -1.8097, -1.7754,  ..., -0.6281, -0.6109, -0.5596],
         [-1.7925, -1.7925, -1.7583,  ..., -0.6794, -0.6965, -0.6794]],

        [[-1.6155, -1.6331, -0.0049,  ..., -0.0049, -0.0049, -0.0049],
         [-1.6331, -1.6155, -1.6506,  ..., -0.0049, -0.0049, -0.0049],
         [-1.5980, -1.5980, -1.6331,  ..., -0.0049, -0.0049, -0.0049],
         ...,
         [-1.6856, -1.6856, -1.6856,  ..., -0.0399, -0.0399, -0.0049],
         [-1.6856, -1.7031, -1.6681,  ..., -0.1099, -0.0749, -0.0574],
         [-1.6856, -1.6856, -1.6506,  ..., -0.1800, -0.1275, -0.1450]],

        [[-1.4907, -1.5081,  0.0082,  ...,  0.0082,  0.0082,  0.0082],
         [-1.5081, -1.4907, -1.5256,  ...,  0.0082,  0.0082,  0.0082],
         [-1.4733, -1.4733, -1.5081,  ...,  0.0082,  0.0082,  0.0082],
         ...,
         [-1.4210, -1.4210, -1.4210,  ...,  0.4962,  0.4962,  0.5311],
         [-1.4210, -1.4384, -1.4036,  ...,  0.4614,  0.4265,  0.4439],
         [-1.4210, -1.4210, -1.3861,  ...,  0.3045,  0.3045,  0.3393]]]), 999)
-------------------我的分割线10----------------------
type(dataset[1281166][0])： <class 'torch.Tensor'>
-------------------我的分割线11----------------------
dataset[1281166][0].shape： torch.Size([3, 224, 224])
-------------------我的分割线12----------------------
dataset[1281166][1]： 999
-------------------结束监视代码----------------------
```
由此可见，PyTorch官方的`datasets.ImageFolder`类能够把图片转换成之后神经网络可用的`<class 'torch.Tensor'>`数据类型。在我的这次训练里，图片被统一缩放成224x224的大小。这个224x224的大小是在`/Swin-Transformer/config.py`文件中的`_C.DATA.IMG_SIZE = 224`这里设定的，并在`def build_transform(is_train, config):`函数里的`transform = create_transform(...)`里被用来初始化`transform(<class 'torchvision.transforms.transforms.Compose'>)`对象。在vscode的全局搜索框里，直接搜索`DATA.IMG_SIZE`，可以发现整个`Swin Transformer`代码都在哪里调用了`_C.DATA.IMG_SIZE`或`config.DATA.IMG_SIZE`。
再来进行下述实验：
``` python
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
    print("-------------------开始监视代码----------------------")
    print(type(dataset.class_to_idx))
    print("-------------------我的分割线1----------------------")
    print(dataset.class_to_idx)
    print("-------------------我的分割线2----------------------")
    print(type(dataset.imgs))
    print("-------------------结束监视代码----------------------")
    exit()
nb_classes = 1000
```
结果为：
```
-------------------开始监视代码----------------------
<class 'dict'>
-------------------我的分割线1----------------------
{'n01440764': 0, 'n01443537': 1, 'n01484850': 2, 'n01491361': 3, 'n01494475': 4, 'n01496331': 5, 'n01498041': 6, 'n01514668': 7, 'n01514859': 8, 'n01518878': 9, 'n01530575': 10, 'n01531178': 11, 'n01532829': 12, 'n01534433': 13, 'n01537544': 14, 'n01558993': 15, 'n01560419': 16, 'n01580077': 17, 'n01582220': 18, 'n01592084': 19, 'n01601694': 20, 'n01608432': 21, 'n01614925': 22, 'n01616318': 23, 'n01622779': 24, 'n01629819': 25, 'n01630670': 26, 'n01631663': 27, 'n01632458': 28, 'n01632777': 29, 'n01641577': 30, 'n01644373': 31, 'n01644900': 32, 'n01664065': 33, 'n01665541': 34, 'n01667114': 35, 'n01667778': 36, 'n01669191': 37, 'n01675722': 38, 'n01677366': 39, 'n01682714': 40, 'n01685808': 41, 'n01687978': 42, 'n01688243': 43, 'n01689811': 44, 'n01692333': 45, 'n01693334': 46, 'n01694178': 47, 'n01695060': 48, 'n01697457': 49, 'n01698640': 50, 'n01704323': 51, 'n01728572': 52, 'n01728920': 53, 'n01729322': 54, 'n01729977': 55, 'n01734418': 56, 'n01735189': 57, 'n01737021': 58, 'n01739381': 59, 'n01740131': 60, 'n01742172': 61, 'n01744401': 62, 'n01748264': 63, 'n01749939': 64, 'n01751748': 65, 'n01753488': 66, 'n01755581': 67, 'n01756291': 68, 'n01768244': 69, 'n01770081': 70, 'n01770393': 71, 'n01773157': 72, 'n01773549': 73, 'n01773797': 74, 'n01774384': 75, 'n01774750': 76, 'n01775062': 77, 'n01776313': 78, 'n01784675': 79, 'n01795545': 80, 'n01796340': 81, 'n01797886': 82, 'n01798484': 83, 'n01806143': 84, 'n01806567': 85, 'n01807496': 86, 'n01817953': 87, 'n01818515': 88, 'n01819313': 89, 'n01820546': 90, 'n01824575': 91, 'n01828970': 92, 'n01829413': 93, 'n01833805': 94, 'n01843065': 95, 'n01843383': 96, 'n01847000': 97, 'n01855032': 98, 'n01855672': 99, 'n01860187': 100, 'n01871265': 101, 'n01872401': 102, 'n01873310': 103, 'n01877812': 104, 'n01882714': 105, 'n01883070': 106, 'n01910747': 107, 'n01914609': 108, 'n01917289': 109, 'n01924916': 110, 'n01930112': 111, 'n01943899': 112, 'n01944390': 113, 'n01945685': 114, 'n01950731': 115, 'n01955084': 116, 'n01968897': 117, 'n01978287': 118, 'n01978455': 119, 'n01980166': 120, 'n01981276': 121, 'n01983481': 122, 'n01984695': 123, 'n01985128': 124, 'n01986214': 125, 'n01990800': 126, 'n02002556': 127, 'n02002724': 128, 'n02006656': 129, 'n02007558': 130, 'n02009229': 131, 'n02009912': 132, 'n02011460': 133, 'n02012849': 134, 'n02013706': 135, 'n02017213': 136, 'n02018207': 137, 'n02018795': 138, 'n02025239': 139, 'n02027492': 140, 'n02028035': 141, 'n02033041': 142, 'n02037110': 143, 'n02051845': 144, 'n02056570': 145, 'n02058221': 146, 'n02066245': 147, 'n02071294': 148, 'n02074367': 149, 'n02077923': 150, 'n02085620': 151, 'n02085782': 152, 'n02085936': 153, 'n02086079': 154, 'n02086240': 155, 'n02086646': 156, 'n02086910': 157, 'n02087046': 158, 'n02087394': 159, 'n02088094': 160, 'n02088238': 161, 'n02088364': 162, 'n02088466': 163, 'n02088632': 164, 'n02089078': 165, 'n02089867': 166, 'n02089973': 167, 'n02090379': 168, 'n02090622': 169, 'n02090721': 170, 'n02091032': 171, 'n02091134': 172, 'n02091244': 173, 'n02091467': 174, 'n02091635': 175, 'n02091831': 176, 'n02092002': 177, 'n02092339': 178, 'n02093256': 179, 'n02093428': 180, 'n02093647': 181, 'n02093754': 182, 'n02093859': 183, 'n02093991': 184, 'n02094114': 185, 'n02094258': 186, 'n02094433': 187, 'n02095314': 188, 'n02095570': 189, 'n02095889': 190, 'n02096051': 191, 'n02096177': 192, 'n02096294': 193, 'n02096437': 194, 'n02096585': 195, 'n02097047': 196, 'n02097130': 197, 'n02097209': 198, 'n02097298': 199, 'n02097474': 200, 'n02097658': 201, 'n02098105': 202, 'n02098286': 203, 'n02098413': 204, 'n02099267': 205, 'n02099429': 206, 'n02099601': 207, 'n02099712': 208, 'n02099849': 209, 'n02100236': 210, 'n02100583': 211, 'n02100735': 212, 'n02100877': 213, 'n02101006': 214, 'n02101388': 215, 'n02101556': 216, 'n02102040': 217, 'n02102177': 218, 'n02102318': 219, 'n02102480': 220, 'n02102973': 221, 'n02104029': 222, 'n02104365': 223, 'n02105056': 224, 'n02105162': 225, 'n02105251': 226, 'n02105412': 227, 'n02105505': 228, 'n02105641': 229, 'n02105855': 230, 'n02106030': 231, 'n02106166': 232, 'n02106382': 233, 'n02106550': 234, 'n02106662': 235, 'n02107142': 236, 'n02107312': 237, 'n02107574': 238, 'n02107683': 239, 'n02107908': 240, 'n02108000': 241, 'n02108089': 242, 'n02108422': 243, 'n02108551': 244, 'n02108915': 245, 'n02109047': 246, 'n02109525': 247, 'n02109961': 248, 'n02110063': 249, 'n02110185': 250, 'n02110341': 251, 'n02110627': 252, 'n02110806': 253, 'n02110958': 254, 'n02111129': 255, 'n02111277': 256, 'n02111500': 257, 'n02111889': 258, 'n02112018': 259, 'n02112137': 260, 'n02112350': 261, 'n02112706': 262, 'n02113023': 263, 'n02113186': 264, 'n02113624': 265, 'n02113712': 266, 'n02113799': 267, 'n02113978': 268, 'n02114367': 269, 'n02114548': 270, 'n02114712': 271, 'n02114855': 272, 'n02115641': 273, 'n02115913': 274, 'n02116738': 275, 'n02117135': 276, 'n02119022': 277, 'n02119789': 278, 'n02120079': 279, 'n02120505': 280, 'n02123045': 281, 'n02123159': 282, 'n02123394': 283, 'n02123597': 284, 'n02124075': 285, 'n02125311': 286, 'n02127052': 287, 'n02128385': 288, 'n02128757': 289, 'n02128925': 290, 'n02129165': 291, 'n02129604': 292, 'n02130308': 293, 'n02132136': 294, 'n02133161': 295, 'n02134084': 296, 'n02134418': 297, 'n02137549': 298, 'n02138441': 299, 'n02165105': 300, 'n02165456': 301, 'n02167151': 302, 'n02168699': 303, 'n02169497': 304, 'n02172182': 305, 'n02174001': 306, 'n02177972': 307, 'n02190166': 308, 'n02206856': 309, 'n02219486': 310, 'n02226429': 311, 'n02229544': 312, 'n02231487': 313, 'n02233338': 314, 'n02236044': 315, 'n02256656': 316, 'n02259212': 317, 'n02264363': 318, 'n02268443': 319, 'n02268853': 320, 'n02276258': 321, 'n02277742': 322, 'n02279972': 323, 'n02280649': 324, 'n02281406': 325, 'n02281787': 326, 'n02317335': 327, 'n02319095': 328, 'n02321529': 329, 'n02325366': 330, 'n02326432': 331, 'n02328150': 332, 'n02342885': 333, 'n02346627': 334, 'n02356798': 335, 'n02361337': 336, 'n02363005': 337, 'n02364673': 338, 'n02389026': 339, 'n02391049': 340, 'n02395406': 341, 'n02396427': 342, 'n02397096': 343, 'n02398521': 344, 'n02403003': 345, 'n02408429': 346, 'n02410509': 347, 'n02412080': 348, 'n02415577': 349, 'n02417914': 350, 'n02422106': 351, 'n02422699': 352, 'n02423022': 353, 'n02437312': 354, 'n02437616': 355, 'n02441942': 356, 'n02442845': 357, 'n02443114': 358, 'n02443484': 359, 'n02444819': 360, 'n02445715': 361, 'n02447366': 362, 'n02454379': 363, 'n02457408': 364, 'n02480495': 365, 'n02480855': 366, 'n02481823': 367, 'n02483362': 368, 'n02483708': 369, 'n02484975': 370, 'n02486261': 371, 'n02486410': 372, 'n02487347': 373, 'n02488291': 374, 'n02488702': 375, 'n02489166': 376, 'n02490219': 377, 'n02492035': 378, 'n02492660': 379, 'n02493509': 380, 'n02493793': 381, 'n02494079': 382, 'n02497673': 383, 'n02500267': 384, 'n02504013': 385, 'n02504458': 386, 'n02509815': 387, 'n02510455': 388, 'n02514041': 389, 'n02526121': 390, 'n02536864': 391, 'n02606052': 392, 'n02607072': 393, 'n02640242': 394, 'n02641379': 395, 'n02643566': 396, 'n02655020': 397, 'n02666196': 398, 'n02667093': 399, 'n02669723': 400, 'n02672831': 401, 'n02676566': 402, 'n02687172': 403, 'n02690373': 404, 'n02692877': 405, 'n02699494': 406, 'n02701002': 407, 'n02704792': 408, 'n02708093': 409, 'n02727426': 410, 'n02730930': 411, 'n02747177': 412, 'n02749479': 413, 'n02769748': 414, 'n02776631': 415, 'n02777292': 416, 'n02782093': 417, 'n02783161': 418, 'n02786058': 419, 'n02787622': 420, 'n02788148': 421, 'n02790996': 422, 'n02791124': 423, 'n02791270': 424, 'n02793495': 425, 'n02794156': 426, 'n02795169': 427, 'n02797295': 428, 'n02799071': 429, 'n02802426': 430, 'n02804414': 431, 'n02804610': 432, 'n02807133': 433, 'n02808304': 434, 'n02808440': 435, 'n02814533': 436, 'n02814860': 437, 'n02815834': 438, 'n02817516': 439, 'n02823428': 440, 'n02823750': 441, 'n02825657': 442, 'n02834397': 443, 'n02835271': 444, 'n02837789': 445, 'n02840245': 446, 'n02841315': 447, 'n02843684': 448, 'n02859443': 449, 'n02860847': 450, 'n02865351': 451, 'n02869837': 452, 'n02870880': 453, 'n02871525': 454, 'n02877765': 455, 'n02879718': 456, 'n02883205': 457, 'n02892201': 458, 'n02892767': 459, 'n02894605': 460, 'n02895154': 461, 'n02906734': 462, 'n02909870': 463, 'n02910353': 464, 'n02916936': 465, 'n02917067': 466, 'n02927161': 467, 'n02930766': 468, 'n02939185': 469, 'n02948072': 470, 'n02950826': 471, 'n02951358': 472, 'n02951585': 473, 'n02963159': 474, 'n02965783': 475, 'n02966193': 476, 'n02966687': 477, 'n02971356': 478, 'n02974003': 479, 'n02977058': 480, 'n02978881': 481, 'n02979186': 482, 'n02980441': 483, 'n02981792': 484, 'n02988304': 485, 'n02992211': 486, 'n02992529': 487, 'n02999410': 488, 'n03000134': 489, 'n03000247': 490, 'n03000684': 491, 'n03014705': 492, 'n03016953': 493, 'n03017168': 494, 'n03018349': 495, 'n03026506': 496, 'n03028079': 497, 'n03032252': 498, 'n03041632': 499, 'n03042490': 500, 'n03045698': 501, 'n03047690': 502, 'n03062245': 503, 'n03063599': 504, 'n03063689': 505, 'n03065424': 506, 'n03075370': 507, 'n03085013': 508, 'n03089624': 509, 'n03095699': 510, 'n03100240': 511, 'n03109150': 512, 'n03110669': 513, 'n03124043': 514, 'n03124170': 515, 'n03125729': 516, 'n03126707': 517, 'n03127747': 518, 'n03127925': 519, 'n03131574': 520, 'n03133878': 521, 'n03134739': 522, 'n03141823': 523, 'n03146219': 524, 'n03160309': 525, 'n03179701': 526, 'n03180011': 527, 'n03187595': 528, 'n03188531': 529, 'n03196217': 530, 'n03197337': 531, 'n03201208': 532, 'n03207743': 533, 'n03207941': 534, 'n03208938': 535, 'n03216828': 536, 'n03218198': 537, 'n03220513': 538, 'n03223299': 539, 'n03240683': 540, 'n03249569': 541, 'n03250847': 542, 'n03255030': 543, 'n03259280': 544, 'n03271574': 545, 'n03272010': 546, 'n03272562': 547, 'n03290653': 548, 'n03291819': 549, 'n03297495': 550, 'n03314780': 551, 'n03325584': 552, 'n03337140': 553, 'n03344393': 554, 'n03345487': 555, 'n03347037': 556, 'n03355925': 557, 'n03372029': 558, 'n03376595': 559, 'n03379051': 560, 'n03384352': 561, 'n03388043': 562, 'n03388183': 563, 'n03388549': 564, 'n03393912': 565, 'n03394916': 566, 'n03400231': 567, 'n03404251': 568, 'n03417042': 569, 'n03424325': 570, 'n03425413': 571, 'n03443371': 572, 'n03444034': 573, 'n03445777': 574, 'n03445924': 575, 'n03447447': 576, 'n03447721': 577, 'n03450230': 578, 'n03452741': 579, 'n03457902': 580, 'n03459775': 581, 'n03461385': 582, 'n03467068': 583, 'n03476684': 584, 'n03476991': 585, 'n03478589': 586, 'n03481172': 587, 'n03482405': 588, 'n03483316': 589, 'n03485407': 590, 'n03485794': 591, 'n03492542': 592, 'n03494278': 593, 'n03495258': 594, 'n03496892': 595, 'n03498962': 596, 'n03527444': 597, 'n03529860': 598, 'n03530642': 599, 'n03532672': 600, 'n03534580': 601, 'n03535780': 602, 'n03538406': 603, 'n03544143': 604, 'n03584254': 605, 'n03584829': 606, 'n03590841': 607, 'n03594734': 608, 'n03594945': 609, 'n03595614': 610, 'n03598930': 611, 'n03599486': 612, 'n03602883': 613, 'n03617480': 614, 'n03623198': 615, 'n03627232': 616, 'n03630383': 617, 'n03633091': 618, 'n03637318': 619, 'n03642806': 620, 'n03649909': 621, 'n03657121': 622, 'n03658185': 623, 'n03661043': 624, 'n03662601': 625, 'n03666591': 626, 'n03670208': 627, 'n03673027': 628, 'n03676483': 629, 'n03680355': 630, 'n03690938': 631, 'n03691459': 632, 'n03692522': 633, 'n03697007': 634, 'n03706229': 635, 'n03709823': 636, 'n03710193': 637, 'n03710637': 638, 'n03710721': 639, 'n03717622': 640, 'n03720891': 641, 'n03721384': 642, 'n03724870': 643, 'n03729826': 644, 'n03733131': 645, 'n03733281': 646, 'n03733805': 647, 'n03742115': 648, 'n03743016': 649, 'n03759954': 650, 'n03761084': 651, 'n03763968': 652, 'n03764736': 653, 'n03769881': 654, 'n03770439': 655, 'n03770679': 656, 'n03773504': 657, 'n03775071': 658, 'n03775546': 659, 'n03776460': 660, 'n03777568': 661, 'n03777754': 662, 'n03781244': 663, 'n03782006': 664, 'n03785016': 665, 'n03786901': 666, 'n03787032': 667, 'n03788195': 668, 'n03788365': 669, 'n03791053': 670, 'n03792782': 671, 'n03792972': 672, 'n03793489': 673, 'n03794056': 674, 'n03796401': 675, 'n03803284': 676, 'n03804744': 677, 'n03814639': 678, 'n03814906': 679, 'n03825788': 680, 'n03832673': 681, 'n03837869': 682, 'n03838899': 683, 'n03840681': 684, 'n03841143': 685, 'n03843555': 686, 'n03854065': 687, 'n03857828': 688, 'n03866082': 689, 'n03868242': 690, 'n03868863': 691, 'n03871628': 692, 'n03873416': 693, 'n03874293': 694, 'n03874599': 695, 'n03876231': 696, 'n03877472': 697, 'n03877845': 698, 'n03884397': 699, 'n03887697': 700, 'n03888257': 701, 'n03888605': 702, 'n03891251': 703, 'n03891332': 704, 'n03895866': 705, 'n03899768': 706, 'n03902125': 707, 'n03903868': 708, 'n03908618': 709, 'n03908714': 710, 'n03916031': 711, 'n03920288': 712, 'n03924679': 713, 'n03929660': 714, 'n03929855': 715, 'n03930313': 716, 'n03930630': 717, 'n03933933': 718, 'n03935335': 719, 'n03937543': 720, 'n03938244': 721, 'n03942813': 722, 'n03944341': 723, 'n03947888': 724, 'n03950228': 725, 'n03954731': 726, 'n03956157': 727, 'n03958227': 728, 'n03961711': 729, 'n03967562': 730, 'n03970156': 731, 'n03976467': 732, 'n03976657': 733, 'n03977966': 734, 'n03980874': 735, 'n03982430': 736, 'n03983396': 737, 'n03991062': 738, 'n03992509': 739, 'n03995372': 740, 'n03998194': 741, 'n04004767': 742, 'n04005630': 743, 'n04008634': 744, 'n04009552': 745, 'n04019541': 746, 'n04023962': 747, 'n04026417': 748, 'n04033901': 749, 'n04033995': 750, 'n04037443': 751, 'n04039381': 752, 'n04040759': 753, 'n04041544': 754, 'n04044716': 755, 'n04049303': 756, 'n04065272': 757, 'n04067472': 758, 'n04069434': 759, 'n04070727': 760, 'n04074963': 761, 'n04081281': 762, 'n04086273': 763, 'n04090263': 764, 'n04099969': 765, 'n04111531': 766, 'n04116512': 767, 'n04118538': 768, 'n04118776': 769, 'n04120489': 770, 'n04125021': 771, 'n04127249': 772, 'n04131690': 773, 'n04133789': 774, 'n04136333': 775, 'n04141076': 776, 'n04141327': 777, 'n04141975': 778, 'n04146614': 779, 'n04147183': 780, 'n04149813': 781, 'n04152593': 782, 'n04153751': 783, 'n04154565': 784, 'n04162706': 785, 'n04179913': 786, 'n04192698': 787, 'n04200800': 788, 'n04201297': 789, 'n04204238': 790, 'n04204347': 791, 'n04208210': 792, 'n04209133': 793, 'n04209239': 794, 'n04228054': 795, 'n04229816': 796, 'n04235860': 797, 'n04238763': 798, 'n04239074': 799, 'n04243546': 800, 'n04251144': 801, 'n04252077': 802, 'n04252225': 803, 'n04254120': 804, 'n04254680': 805, 'n04254777': 806, 'n04258138': 807, 'n04259630': 808, 'n04263257': 809, 'n04264628': 810, 'n04265275': 811, 'n04266014': 812, 'n04270147': 813, 'n04273569': 814, 'n04275548': 815, 'n04277352': 816, 'n04285008': 817, 'n04286575': 818, 'n04296562': 819, 'n04310018': 820, 'n04311004': 821, 'n04311174': 822, 'n04317175': 823, 'n04325704': 824, 'n04326547': 825, 'n04328186': 826, 'n04330267': 827, 'n04332243': 828, 'n04335435': 829, 'n04336792': 830, 'n04344873': 831, 'n04346328': 832, 'n04347754': 833, 'n04350905': 834, 'n04355338': 835, 'n04355933': 836, 'n04356056': 837, 'n04357314': 838, 'n04366367': 839, 'n04367480': 840, 'n04370456': 841, 'n04371430': 842, 'n04371774': 843, 'n04372370': 844, 'n04376876': 845, 'n04380533': 846, 'n04389033': 847, 'n04392985': 848, 'n04398044': 849, 'n04399382': 850, 'n04404412': 851, 'n04409515': 852, 'n04417672': 853, 'n04418357': 854, 'n04423845': 855, 'n04428191': 856, 'n04429376': 857, 'n04435653': 858, 'n04442312': 859, 'n04443257': 860, 'n04447861': 861, 'n04456115': 862, 'n04458633': 863, 'n04461696': 864, 'n04462240': 865, 'n04465501': 866, 'n04467665': 867, 'n04476259': 868, 'n04479046': 869, 'n04482393': 870, 'n04483307': 871, 'n04485082': 872, 'n04486054': 873, 'n04487081': 874, 'n04487394': 875, 'n04493381': 876, 'n04501370': 877, 'n04505470': 878, 'n04507155': 879, 'n04509417': 880, 'n04515003': 881, 'n04517823': 882, 'n04522168': 883, 'n04523525': 884, 'n04525038': 885, 'n04525305': 886, 'n04532106': 887, 'n04532670': 888, 'n04536866': 889, 'n04540053': 890, 'n04542943': 891, 'n04548280': 892, 'n04548362': 893, 'n04550184': 894, 'n04552348': 895, 'n04553703': 896, 'n04554684': 897, 'n04557648': 898, 'n04560804': 899, 'n04562935': 900, 'n04579145': 901, 'n04579432': 902, 'n04584207': 903, 'n04589890': 904, 'n04590129': 905, 'n04591157': 906, 'n04591713': 907, 'n04592741': 908, 'n04596742': 909, 'n04597913': 910, 'n04599235': 911, 'n04604644': 912, 'n04606251': 913, 'n04612504': 914, 'n04613696': 915, 'n06359193': 916, 'n06596364': 917, 'n06785654': 918, 'n06794110': 919, 'n06874185': 920, 'n07248320': 921, 'n07565083': 922, 'n07579787': 923, 'n07583066': 924, 'n07584110': 925, 'n07590611': 926, 'n07613480': 927, 'n07614500': 928, 'n07615774': 929, 'n07684084': 930, 'n07693725': 931, 'n07695742': 932, 'n07697313': 933, 'n07697537': 934, 'n07711569': 935, 'n07714571': 936, 'n07714990': 937, 'n07715103': 938, 'n07716358': 939, 'n07716906': 940, 'n07717410': 941, 'n07717556': 942, 'n07718472': 943, 'n07718747': 944, 'n07720875': 945, 'n07730033': 946, 'n07734744': 947, 'n07742313': 948, 'n07745940': 949, 'n07747607': 950, 'n07749582': 951, 'n07753113': 952, 'n07753275': 953, 'n07753592': 954, 'n07754684': 955, 'n07760859': 956, 'n07768694': 957, 'n07802026': 958, 'n07831146': 959, 'n07836838': 960, 'n07860988': 961, 'n07871810': 962, 'n07873807': 963, 'n07875152': 964, 'n07880968': 965, 'n07892512': 966, 'n07920052': 967, 'n07930864': 968, 'n07932039': 969, 'n09193705': 970, 'n09229709': 971, 'n09246464': 972, 'n09256479': 973, 'n09288635': 974, 'n09332890': 975, 'n09399592': 976, 'n09421951': 977, 'n09428293': 978, 'n09468604': 979, 'n09472597': 980, 'n09835506': 981, 'n10148035': 982, 'n10565667': 983, 'n11879895': 984, 'n11939491': 985, 'n12057211': 986, 'n12144580': 987, 'n12267677': 988, 'n12620546': 989, 'n12768682': 990, 'n12985857': 991, 'n12998815': 992, 'n13037406': 993, 'n13040303': 994, 'n13044778': 995, 'n13052670': 996, 'n13054560': 997, 'n13133613': 998, 'n15075141': 999}
-------------------我的分割线2----------------------
<class 'list'>
-------------------结束监视代码----------------------
```
（注意：之所以没有直接`print(dataset.imgs)`，是因为把这个print出来，内容太多了。稍后细说。）
由此我们就明白了，`dataset.class_to_idx`的类型是`<class 'dict'>`，是一个字典。我们再看它的内容可知，`dataset.class_to_idx`是把每个类别的名称（注意，在我用的数据集里，类别名称并不是用英语来表示的，而是用一个n开头的代号来表示的。这一点之前已经分析过了，对于人类的理解会有负面作用，但不影响网络的训练。）转换成一个从0开始的整数，我们权且称之为类别编号。我们看到类别编号从0排到999，这说明，我用的数据集是ImageNet-1k数据集。这个数据集有1000个物体类别。
下面我们必须仔细地研究一下`dataset.imgs`这个对象了。这个对象的类型是`<class 'list'>`。我们试运行下述代码：
``` python
else:
    root = os.path.join(config.DATA.DATA_PATH, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
    print("-------------------开始监视代码----------------------")
    print(type(dataset.imgs))
    print("-------------------我的分割线1----------------------")
    print(len(dataset.imgs))
    print("-------------------我的分割线2----------------------")
    print(type(dataset.imgs[0]))
    print("-------------------我的分割线3----------------------")
    print(dataset.imgs[0])
    print("-------------------我的分割线4----------------------")
    print(type(dataset.imgs[1281166]))
    print("-------------------我的分割线5----------------------")
    print(dataset.imgs[1281166])
    print("-------------------结束监视代码----------------------")
    exit()
nb_classes = 1000
```
结果为：
```
-------------------开始监视代码----------------------
<class 'list'>
-------------------我的分割线1----------------------
1281167
-------------------我的分割线2----------------------
<class 'tuple'>
-------------------我的分割线3----------------------
('/XXX-XXX/XXX_XXX_data/imagenet/origin/train/n01440764/n01440764_10026.JPEG', 0)
-------------------我的分割线4----------------------
<class 'tuple'>
-------------------我的分割线5----------------------
('/XXX-XXX/XXX_XXX_data/imagenet/origin/train/n15075141/n15075141_9993.JPEG', 999)
-------------------结束监视代码----------------------
```
至此，我们对于`dataset = datasets.ImageFolder(root, transform=transform)`这行代码输出的`dataset`变量已经有了非常详细的认识。`dataset.imgs`对象的类型是`<class 'list'>`，是一个列表。这个列表的长度是`1281167`，这表明训练集一共有1281167多张图片。我们把第一张图片和最后一张图片对应的索引print出来，可以看到，`dataset.imgs`这个列表里面装的是一些元组，每张图片被写成了一个元组。每个元组（也就是对应着每张图片）由两个分量构成：第一个分量是图片所在的绝对路径；第二个分量是图片所属的类别编号（0-999）。这样的数据结构为接下来使用这些训练数据做好了相应准备。（下面很快就可以看到，`dataset = datasets.ImageFolder(root, transform=transform)`这行代码输出的`dataset`变量将被用来初始化训练用`DataLoader`。）

至此，`build_dataset(is_train, config)`这个函数执行完毕。这个函数返回了`dataset`和`nb_classes`这两个对象。其中`nb_classes`就是整数`1000`。我们回到`build_loader(config)`这个函数的代码：
``` python
def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
        is_train=True, config=config
    )
    config.freeze()
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset"
    )
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(
        f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset"
    )
```
按照之前的分析，我们已经弄清楚了这一行：
``` python
dataset_train, config.MODEL.NUM_CLASSES = build_dataset(
    is_train=True, config=config
)
```
的执行细节。这行代码返回的`dataset_train`对象应该是一个`<class 'torchvision.datasets.folder.ImageFolder'>`类型的对象。这个对象在使用时，调用它的`dataset_train.imgs`属性，可以得到一个列表。这个列表里面装的是很多元组，每一个元组由图片的绝对路径和图片的类别编号这两部分组成。`dataset_train.class_to_idx`属性是一个字典，这个字典把图片类别名称（在我的数据中是一个编号，而不是英文名称）赋予一个0-999的类别编号。config.MODEL.NUM_CLASSES这个参数被设置成了`1000`。这个是因为我使用的是`ImageNet-1k`数据集。按照完全相同的方式，构造了验证集对象`dataset_val`。这个对象的使用方法和`dataset_train`完全一样。`dist.get_rank()`还是和分布式训练有关。以后再详细研究。

接下来进入这两行代码：
``` python
num_tasks = dist.get_world_size()
global_rank = dist.get_rank()
```
我们来试运行下述代码：
``` python
num_tasks = dist.get_world_size()
global_rank = dist.get_rank()
print("-------------------开始监视代码----------------------")
print("num_tasks：", num_tasks)
print("-------------------我的分割线1----------------------")
print("global_rank：", global_rank)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
num_tasks： 1
-------------------我的分割线1----------------------
global_rank： 0
-------------------结束监视代码----------------------
```
这两个常数的具体含义和用法，留待以后分析。我猜想，可能还是用于分布式训练。

再试运行下述代码：
``` python
print("-------------------开始监视代码----------------------")
print(config.DATA.ZIP_MODE)
print("-------------------我的分割线1----------------------")
print(config.DATA.CACHE_MODE == "part")
print("-------------------结束监视代码----------------------")
exit()
if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
    indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
    sampler_train = SubsetRandomSampler(indices)
else:
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
```
结果为：
```
-------------------开始监视代码----------------------
False
-------------------我的分割线1----------------------
True
-------------------结束监视代码----------------------
```

(注：我再补充一个用来探测`if-else`语句执行过程的方法：
``` python
if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == "part":
    print("执行了这里1")
    exit()
    indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
    sampler_train = SubsetRandomSampler(indices)
else:
    print("执行了这里2")
    exit()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
```
这个方法也可以很容易地判断出，程序执行的是条件选择语句的哪个分句。)

因此，上述语句会执行`else:`的部分，意即，接下来会执行下述代码：
``` python
else:
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
```
这段代码的核心组件是`torch.utils.data.DistributedSampler`类。这个类的官方文档参见[这里](https://pytorch.org/docs/1.7.1/data.html?highlight=distributedsampler#torch.utils.data.distributed.DistributedSampler)。关于Pytorch多机多卡分布式训练的一个有用的教程，参见[这里](https://zhuanlan.zhihu.com/p/68717029)。我们来试运行下述代码：
``` python
else:
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("-------------------开始监视代码----------------------")
    print(type(sampler_train))
    print("-------------------我的分割线1----------------------")
    print(sampler_train)
    print("-------------------结束监视代码----------------------")
    exit()
```
结果为：
``` 
-------------------开始监视代码----------------------
<class 'torch.utils.data.distributed.DistributedSampler'>
-------------------我的分割线1----------------------
<torch.utils.data.distributed.DistributedSampler object at 0x7fb4ba0ad7d0>
-------------------结束监视代码----------------------
```
关于这个分布式训练模块的用法，之后再详细研究。根据[这个教程](https://zhuanlan.zhihu.com/p/68717029)，以下几个参数的含义是：world size指进程总数，在这里就是我们使用的卡数；rank指进程序号，local_rank指本地序号，两者的区别在于前者用于进程间通讯，后者用于本地设备分配。关于这些含义的详细理解，以后再补全。

接下来是这样的两行代码：
``` python
indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
sampler_val = SubsetRandomSampler(indices)
```
我要再一次对这两行代码进行庖丁解牛。首先，我们来看看第一行代码的输入参数都是什么。试运行下述代码：
``` python
print("-------------------开始监视代码----------------------")
print("dist.get_rank()：", dist.get_rank())
print("-------------------我的分割线1----------------------")
print("len(dataset_val)：", len(dataset_val))
print("-------------------我的分割线2----------------------")
print("dist.get_world_size()：", dist.get_world_size())
print("-------------------结束监视代码----------------------")
exit()
indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
sampler_val = SubsetRandomSampler(indices)
```
结果为：
```
-------------------开始监视代码----------------------
dist.get_rank()： 0
-------------------我的分割线1----------------------
len(dataset_val)： 50000
-------------------我的分割线2----------------------
dist.get_world_size()： 1
-------------------结束监视代码----------------------
```
由此，我知道了`indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())`这行代码的三个输入参数，分别是`0`，`50000`和`1`。
接下来试运行下述代码：
``` python
indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
print("-------------------开始监视代码----------------------")
print(type(indices))
print("-------------------我的分割线1----------------------")
print(indices)
print("-------------------我的分割线2----------------------")
print(len(indices) == 50000)
print("-------------------结束监视代码----------------------")
exit()
sampler_val = SubsetRandomSampler(indices)
```
结果为：
```
-------------------开始监视代码----------------------
<class 'numpy.ndarray'>
-------------------我的分割线1----------------------
[    0     1     2 ... 49997 49998 49999]
-------------------我的分割线2----------------------
True
-------------------结束监视代码----------------------
```
由此知：`indices`变量是一个`<class 'numpy.ndarray'>`类型的变量，相当于一个一维数组。`indices`变量的长度是`50000`。这个`indices`变量被用来初始化下面一行的那个`SubsetRandomSampler`类的实例。试运行下述代码：
``` python
indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
sampler_val = SubsetRandomSampler(indices)
print("-------------------开始监视代码----------------------")
print(type(sampler_val))
print("-------------------我的分割线1----------------------")
print(sampler_val)
print("-------------------结束监视代码----------------------")
exit()
```
结果为：
```
-------------------开始监视代码----------------------
<class 'data.samplers.SubsetRandomSampler'>
-------------------我的分割线1----------------------
<data.samplers.SubsetRandomSampler object at 0x7fab4098e490>
-------------------结束监视代码----------------------
```
这个`SubsetRandomSampler`类的定义在`/Swin-Transformer/data/samplers.py`文件里。这个python文件里只有这个`SubsetRandomSampler`类的定义。下面的代码就是`/Swin-Transformer/data/samplers.py`这个文件的全部代码：
``` python
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.utils.data


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
```
可以看到，`SubsetRandomSampler`类的核心内容是继承了父类`torch.utils.data.Sampler`的内容，然后在这个`torch.utils.data.Sampler`父类的基础上来实现。这段代码里用到了一个`torch.randperm`函数，我们来测试一下`torch.randperm`函数的用法。下面的代码需要在一个空白的python文件里来运行：
``` python
import torch

a = torch.randperm(5)
b = torch.randperm(5)
c = torch.randperm(5)

print(a)
print(b)
print(c)
```
结果为（注：结果每次都不一样，是随机的）：
```
tensor([0, 3, 4, 1, 2])
tensor([2, 0, 3, 1, 4])
tensor([3, 4, 2, 0, 1])
```
`torch.randperm`函数的目的是：生成一个给定长度的随机一维张量，参见[torch.randperm官方文档](https://pytorch.org/docs/1.7.1/generated/torch.randperm.html#torch.randperm)。

我们进入PyTorch源码中`torch.utils.data.Sampler`类的部分：
``` python
class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError()`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising an `NotImplementedError` will propagate and and make the call
    #     fail where it could have use `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)
```
阅读一下PyTorch官方对于`torch.utils.data.Sampler`类的注释部分可知，`torch.utils.data.Sampler`类是所有`Samplers`的基类，每个`Sampler`子类都必须实现一个`__iter__`方法和一个`__len__`方法。`__iter__`方法提供了一种在数据集元素的索引上进行迭代遍历的方法；`__len__`方法返回了返回的迭代器的长度。下面还有注意事项：`__len__`方法并不是`torch.utils.data.DataLoader`类所必须的方法，但是建议在任何需要计算`torch.utils.data.DataLoader`的长度的场合实现`__len__`方法。（注：我在这里直接把英文注释翻译过来了，可能确实有些拗口。之后我再通过代码的分析来详细体会这里的含义。）

接下来我们回到`/Swin-Transformer/data/build.py`文件的`build_loader(config)`函数。我们来看下面的代码：
``` python
data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    sampler=sampler_train,
    batch_size=config.DATA.BATCH_SIZE,
    num_workers=config.DATA.NUM_WORKERS,
    pin_memory=config.DATA.PIN_MEMORY,
    drop_last=True,
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    sampler=sampler_val,
    batch_size=config.DATA.BATCH_SIZE,
    shuffle=False,
    num_workers=config.DATA.NUM_WORKERS,
    pin_memory=config.DATA.PIN_MEMORY,
    drop_last=False,
)
```
这两行代码就是初始化`DataLoader`的核心代码。我们首先来关注训练用`DataLoader`：
``` python
data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    sampler=sampler_train,
    batch_size=config.DATA.BATCH_SIZE,
    num_workers=config.DATA.NUM_WORKERS,
    pin_memory=config.DATA.PIN_MEMORY,
    drop_last=True,
)
```
我希望，通过这次Swin Transformer代码的细致剖析，能够学会该如何写深度学习的训练代码和数据接口。所以，有必要仔细地剖析一下`DataLoader`的初始化代码。关于`torch.utils.data.DataLoader`这个类，可以参考[PyTorch官方torch.utils.data.DataLoader文档](https://pytorch.org/docs/1.7.1/data.html#torch.utils.data.DataLoader)。按照惯例，我们还是先来看看输入到`torch.utils.data.DataLoader`里的都是什么。首先来看训练用`Dataloader`。试运行下述代码：
``` python
print("----------------------开始监视代码----------------------")
print("type(dataset_train)：", type(dataset_train))
print("----------------------我的分割线1----------------------")
print("dataset_train：", dataset_train)
print("----------------------我的分割线2----------------------")
print("type(sampler_train)：", type(sampler_train))
print("----------------------我的分割线3----------------------")
print("sampler_train：", sampler_train)
print("----------------------我的分割线4----------------------")
print("type(config.DATA.BATCH_SIZE)：", type(config.DATA.BATCH_SIZE))
print("----------------------我的分割线5----------------------")
print("config.DATA.BATCH_SIZE：", config.DATA.BATCH_SIZE)
print("----------------------我的分割线6----------------------")
print("type(config.DATA.NUM_WORKERS)：", type(config.DATA.NUM_WORKERS))
print("----------------------我的分割线7----------------------")
print("config.DATA.NUM_WORKERS：", config.DATA.NUM_WORKERS)
print("----------------------我的分割线8----------------------")
print("type(config.DATA.PIN_MEMORY)：", type(config.DATA.PIN_MEMORY))
print("----------------------我的分割线9----------------------")
print("config.DATA.PIN_MEMORY：", config.DATA.PIN_MEMORY)
print("----------------------结束监视代码----------------------")
exit()
data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    sampler=sampler_train,
    batch_size=config.DATA.BATCH_SIZE,
    num_workers=config.DATA.NUM_WORKERS,
    pin_memory=config.DATA.PIN_MEMORY,
    drop_last=True,
)
```
结果为：
```
----------------------开始监视代码----------------------
type(dataset_train)： <class 'torchvision.datasets.folder.ImageFolder'>
----------------------我的分割线1----------------------
dataset_train： Dataset ImageFolder
    Number of datapoints: 1281167
    Root location: /XXX-XXX/XXX_XXX_data/imagenet/origin/train
    StandardTransform
Transform: Compose(
               RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC)
               RandomXXXtalFlip(p=0.5)
               <timm.data.auto_augment.RandAugment object at 0x7f7f5950d710>
               ToTensor()
               Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
               <timm.data.random_erasing.RandomErasing object at 0x7f7f58f4ce90>
           )
----------------------我的分割线2----------------------
type(sampler_train)： <class 'torch.utils.data.distributed.DistributedSampler'>
----------------------我的分割线3----------------------
sampler_train： <torch.utils.data.distributed.DistributedSampler object at 0x7f7f58f4c090>
----------------------我的分割线4----------------------
type(config.DATA.BATCH_SIZE)： <class 'int'>
----------------------我的分割线5----------------------
config.DATA.BATCH_SIZE： 32
----------------------我的分割线6----------------------
type(config.DATA.NUM_WORKERS)： <class 'int'>
----------------------我的分割线7----------------------
config.DATA.NUM_WORKERS： 8
----------------------我的分割线8----------------------
type(config.DATA.PIN_MEMORY)： <class 'bool'>
----------------------我的分割线9----------------------
config.DATA.PIN_MEMORY： True
----------------------结束监视代码----------------------
```
参照[PyTorch官方torch.utils.data.DataLoader文档](https://pytorch.org/docs/1.7.1/data.html#torch.utils.data.DataLoader)以及上面print出来的内容，我们来分析一下初始化一个`torch.utils.data.DataLoader`类的实例时需要提供哪些输入。要初始化一个`torch.utils.data.DataLoader`类的实例，需要提供如下这些输入：
```
dataset: 必须参数。是一个torch.utils.data.dataset.Dataset[T_co]类型的变量，提供了一个数据集，从这个数据集中加载数据。
batch_size: 可选参数。每个批次要加载多少个数据样本，默认值是1。
shuffle: 可选参数。设为True则意味着，在每个epoch时都对数据进行随机重排。默认值是False。
sampler: 可选参数。是一个torch.utils.data.sampler.Sampler[int]类型的变量。默认值是None。
batch_sampler: 可选参数。是一个torch.utils.data.sampler.Sampler[Sequence[int]]类型的变量。和sampler一样，但一次返回一批索引。这个参数不能和batch_size、shuffle、sampler 和 drop_last同时使用。默认值是None。
num_workers: 可选参数。这个参数的含义是：使用多少个子进程来加载数据。num_workers取0意味着数据将在主进程中加载。默认值是0。
collate_fn: 可选参数。是一个Callable[List[T], Any]类型的变量。合并一个样本列表，构造出一个小批量张量。当要从一个映射类型的数据集中批量加载数据时，使用这个参数。默认值是None。
pin_memory: 可选参数。如果设为True，则数据加载器在返回张量之前，会先把张量复制到CUDA的针脚内存中。如果你的数据元素的类型是自定义的数据类型，或者你的collate_fn返回的是一个自定义数据类型的小批量张量，则要参看PyTorch官方文档提供的相关例子。默认值是False。
drop_last: 可选参数。如果设为True，则当batch size不能整除dataset size时，会丢弃掉最后的那个不完整的数据批次。如果设为False并且batch size不能整除dataset size，则最后的一个数据批次的大小就会，略小。默认值是False。
（剩下的参数，以后需要时再详细学习。）
```
所以，根据上述PyTorch官方文档中`torch.utils.data.DataLoader`的相关介绍，可以看出，在`Swin Transformer`的训练用`DataLoader`实例初始化的时候，论文作者提供了`dataset`，`sampler`，`batch_size`，`num_workers`，`pin_memory`和`drop_last`这些参数的值。`dataset`参数被赋予了一个`<class 'torchvision.datasets.folder.ImageFolder'>`类型的数据集，`sampler`参数被赋予了一个`<class 'torch.utils.data.distributed.DistributedSampler'>`类型的采样子。`batch_size`设为`32`，`num_workers`设为`8`，`pin_memory`设为`True`，`drop_last`设为`True`（意即，丢弃掉最后的那个不完整的批次）。至此，我们完整地分析了`Swin Transformer`训练用DataLoader初始化时，都提供了哪些参数。`Swin Transformer`的训练用DataLoader在初始化时，接收了数据集、采样器、批次大小、进程数、是否使用针孔内存、是否丢弃掉最后的不完整数据批次这些参数。将这些参数送入`torch.utils.data.DataLoader`类的构造函数，就初始化了一个训练用`DataLoader`的实例。我们再来看看初始化好的这个训练用`DataLoader`的实例到底是什么样子的。试运行下述代码：
``` python
data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    sampler=sampler_train,
    batch_size=config.DATA.BATCH_SIZE,
    num_workers=config.DATA.NUM_WORKERS,
    pin_memory=config.DATA.PIN_MEMORY,
    drop_last=True,
)
print("----------------------开始监视代码----------------------")
print("type(data_loader_train)：", type(data_loader_train))
print("----------------------我的分割线1----------------------")
print("data_loader_train：", data_loader_train)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
``` 
----------------------开始监视代码----------------------
type(data_loader_train)： <class 'torch.utils.data.dataloader.DataLoader'>
----------------------我的分割线1----------------------
data_loader_train： <torch.utils.data.dataloader.DataLoader object at 0x7f8dc8cd1d50>
----------------------结束监视代码----------------------
```
单纯这样的print，看不出来什么实质性的内容。关于`DataLoader`对象的用法，以后用到时再细看。

下面就是验证集`DataLoader`的构造：
``` python
data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    sampler=sampler_val,
    batch_size=config.DATA.BATCH_SIZE,
    shuffle=False,
    num_workers=config.DATA.NUM_WORKERS,
    pin_memory=config.DATA.PIN_MEMORY,
    drop_last=False,
)
```
验证集`DataLoader`的构造和训练集`DataLoader`的构造几乎完全一样。差别在于：验证集`DataLoader`的构造把`shuffle`设为了`False`，`drop_last`设为了`False`。这就是说，在验证阶段，不需要对数据进行随机重排，也不会丢弃掉最后的不完整批次。

看完了`Swin Transformer`代码的训练集`DataLoader`和验证集`DataLoader`这两个`DataLoader`的构造，我们会发现：在初始化一个`DataLoader`时，最主要的部分是要提供`DataLoader`构造函数所需的这两个参数：
```
dataset: 必须参数。是一个torch.utils.data.dataset.Dataset[T_co]类型的变量，提供了一个数据集，从这个数据集中加载数据。
sampler: 可选参数。是一个torch.utils.data.sampler.Sampler[int]类型的变量。默认值是None。
```
这两个参数的代码，是需要结合数据集的特点以及多卡训练的要求来自己写的。所以，要写出一个完整的`DataLoader`，最重要的部分就是写出`dataset`和`sampler`这两个变量所对应的类的各个函数。虽然可能上面已经有了一些不完整的分析，不过我们在这里再一次详细地回顾一下，`Swin Transformer`在编写`ImageNet`数据接口的时候，是如何实现初始化`DataLoader`所需的`dataset`变量和`sampler`变量的。
我们以训练用`DataLoader`为例来分析。训练用`DataLoader`的初始化代码如下：
``` python
data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    sampler=sampler_train,
    batch_size=config.DATA.BATCH_SIZE,
    num_workers=config.DATA.NUM_WORKERS,
    pin_memory=config.DATA.PIN_MEMORY,
    drop_last=True,
)
```
我们利用`vscode`的代码跳转功能，进入到`dataset_train`的定义里面，经过多次进入和跳转（此处省略了详细的跳转步骤），可以发现，`dataset_train`是被`def build_dataset(is_train, config):`函数里的下面这句话初始化的：
``` python
dataset = datasets.ImageFolder(root, transform=transform)
```
注意，对这行代码接收的输入和产生的输出，之前已经进行了一些比较细致的分析，参见[这里](#dataset_first_analysis)。这一次，我们来详细地看看`datasets.ImageFolder`这个类的定义本身。
`datasets.ImageFolder`类是PyTorch官方写好的一个类（参见[这里](https://pytorch.org/vision/0.8/datasets.html#imagefolder)），意即，这个类不是`Swin Transformer`论文作者写的。我们利用`vscode`的定义跳转功能，进入`datasets.ImageFolder`类的定义，会看到如下的代码（注意：下面这个`datasets.ImageFolder`类的定义代码来自文件`/data/XXX/conda_env/swin/lib/python3.7/site-packages/torchvision/datasets/folder.py`，在`vscode`文件上右击鼠标copy path即可）：
``` python
class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
```
参考[文档](https://pytorch.org/vision/0.8/datasets.html#imagefolder)中的相关内容可知，`datasets.ImageFolder`类是一个通用的（或者叫泛型的）数据类，它用来封装这种类型的数据：
```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```
而根据之前已经说过的，我使用的`ImageNet`数据集的目录树组织形式如下：
```
/XXX-XXX/XXX_XXX_data/imagenet/origin
├── train
│   ├── n02033041
│   │   ├── n02033041_10047.JPEG
│   │   ├── n02033041_1933.JPEG
│   │   └── ...
│   ├── n02124075
│   │   ├── n02124075_10012.JPEG
│   │   └── ...
│   └── ...
└── val
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   └── ...
    ├── n01990800
    │   ├── ILSVRC2012_val_00000669.JPEG
    │   └── ...
    └── ...
```
完全就是`datasets.ImageFolder`类所要封装的数据类型的组织形式。因此，只需把`/XXX-XXX/XXX_XXX_data/imagenet/origin/train`作为`root`变量的值传入`datasets.ImageFolder`类的构造函数，就可以顺利地初始化一个`datasets.ImageFolder`类封装的数据的实例了。
`datasets.ImageFolder`类的初始化需要下面的这些参数：
```
root：必须参数。是一个字符串类型的变量，提供了数据集的根路径。在我的训练中，root的值为'/XXX-XXX/XXX_XXX_data/imagenet/origin/train'。在这个根路径的下面，是所有的类别组成的文件夹，每个类别的代号或名字就是文件类的名字。类别文件夹里面就是这个类别的各个不同图片。
transform：可选参数。默认值是None。要求这个参数必须是可调用的（关于可调用性的解释，参考https://www.runoob.com/python/python-func-callable.html）。这个参数接收一个函数或者变换，这个变换能够把一个PIL图像变成一个变换对象。比如在我的训练中，就把一个<class 'torchvision.transforms.transforms.Compose'>类型的对象传给了transform变量。
target_transform：可选参数。默认值是None。要求这个参数必须是可调用的。这个参数接收一个函数或者变换，这个变换能对一个目标进行变换。
loader：可选参数。默认值是<function default_loader>。要求这个参数必须是可调用的。这个函数能够使用一个路径来加载一张图片。
is_valid_file：可选参数。默认值是None。这个变量接收一个函数。这个函数能检查一个图像路径是否是一个有效的文件（用于检查文件是否损坏）。
```
`datasets.ImageFolder`类的核心源码都在它所继承的父类`DatasetFolder`父类里。为了简便起见，我们就不再详细分析`DatasetFolder`父类的源码了。（以后如果有机会详细阅读PyTorch的源码的话，或许会写相应的学习笔记来分析吧。）这里我们只需知道：`datasets.ImageFolder`类是PyTorch官方定义的，用来封装`根路径/类别文件夹/每个类别的不同图片.jpeg/png`类型的数据类型，并且它的用法参见[之前的分析](#datasets.ImageFolder_shili)就够了。

我们再利用vscode的代码跳转功能，进入到初始化训练用`DataLoader`时提供给`sampler`的`sampler_train`的定义里面，我们可以看到，`sampler_train`是由这句话来定义的：
``` python
sampler_train = torch.utils.data.DistributedSampler(
    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
)
```
从这句话可以看到，要初始化一个用于单机多卡训练的`sampler`，还是要把训练数据`dataset_train`、两个整数`num_tasks`和`global_rank`提供给`torch.utils.data.DistributedSampler`类的构造函数。我们由此就明白了，在`Swin Transformer`的训练用`DataLoader`的构造中，最核心的部分就是要写好一个`dataset（torch.utils.data.dataset.Dataset[T_co]类型）`的变量。这个`dataset`写好以后，就可以用它来初始化`sampler`和`DataLoader`了。当然，以后遇到更复杂的模型，不排除可能需要写更复杂的`sampler`。但是，以后如果遇到了更复杂的`sampler`再说。目前只需知道，在`Swin Transformer`中，`sampler_train`的初始化其实很简单，就是把已经构造出来的`dataset_train`送到PyTorch官方写好的`torch.utils.data.DistributedSampler`里，就够了。据此，我们就算是完整且细致地分析了训练用`DataLoader`的初始化所需要知道的一切。
验证用`DataLoader` `data_loader_val`变量的初始化，和训练用`DataLoader`的初始化几乎完全一样，此处就不再展开分析了。
我们继续回到`def build_loader(config):`函数，在初始化了训练和验证用`DataLoader`以后，是下面的代码：
``` python
# setup mixup / cutmix
mixup_fn = None
mixup_active = (
    config.AUG.MIXUP > 0
    or config.AUG.CUTMIX > 0.0
    or config.AUG.CUTMIX_MINMAX is not None
)
```
对这段代码，我还是比较陌生的，此时此刻我还不知道`mixup_fn`和`mixup_active`这两个变量之后会有什么作用。可以看出，`mixup_active`是一个逻辑值。我先来简单地用之前常规的测试方法测试一下吧。试运行下述代码：
``` python
# setup mixup / cutmix
mixup_fn = None
mixup_active = (
    config.AUG.MIXUP > 0
    or config.AUG.CUTMIX > 0.0
    or config.AUG.CUTMIX_MINMAX is not None
)
print("----------------------开始监视代码----------------------")
print("type(mixup_active)：", type(mixup_active))
print("----------------------我的分割线1----------------------")
print("mixup_active：", mixup_active)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(mixup_active)： <class 'bool'>
----------------------我的分割线1----------------------
mixup_active： True
----------------------结束监视代码----------------------
```
注意：`mixup_active`的值为`True`。之后要留意这个逻辑值起了什么作用。
下面的代码是这样的：
``` python
if mixup_active:
    mixup_fn = Mixup(
        mixup_alpha=config.AUG.MIXUP,
        cutmix_alpha=config.AUG.CUTMIX,
        cutmix_minmax=config.AUG.CUTMIX_MINMAX,
        prob=config.AUG.MIXUP_PROB,
        switch_prob=config.AUG.MIXUP_SWITCH_PROB,
        mode=config.AUG.MIXUP_MODE,
        label_smoothing=config.MODEL.LABEL_SMOOTHING,
        num_classes=config.MODEL.NUM_CLASSES,
    )
```
由于`mixup_active`的值为`True`，因此会执行这个`if`语句所包裹的代码。也就是说，会使用`Mixup`类的构造函数和一些已知参数初始化一个`Mixup`类的实例，并把这个`Mixup`类的实例赋值给`mixup_fn`变量。这里最核心的是`Mixup`类的定义，这个类是在`/Swin-Transformer/data/build.py`的开头由`from timm.data import Mixup`这句话引入的。看来，`Mixup`类并不是`Swin Transformer`代码的一部分，而是`timm`这个第三方Python库的一部分。`Mixup`类的完整定义位于`/data/XXX/conda_env/swin/lib/python3.7/site-packages/timm/data/mixup.py`文件中，它的完整定义我们就不展示了。但是我们要弄明白`Mixup`类的实例的功能和用法。参考这个[Python timm库教程](https://blog.csdn.net/qq_41917697/article/details/115026308)和[timm的Github官方页面](https://github.com/rwightman/pytorch-image-models)，可以知道，`timm`库是一个集成了许多SOTA模型的库，具体用法以后再详细看。我们先来验证一下下述代码：
``` python
if mixup_active:
    mixup_fn = Mixup(
        mixup_alpha=config.AUG.MIXUP,
        cutmix_alpha=config.AUG.CUTMIX,
        cutmix_minmax=config.AUG.CUTMIX_MINMAX,
        prob=config.AUG.MIXUP_PROB,
        switch_prob=config.AUG.MIXUP_SWITCH_PROB,
        mode=config.AUG.MIXUP_MODE,
        label_smoothing=config.MODEL.LABEL_SMOOTHING,
        num_classes=config.MODEL.NUM_CLASSES,
    )
    print("----------------------开始监视代码----------------------")
    print("type(mixup_fn)：", type(mixup_fn))
    print("----------------------我的分割线1----------------------")
    print("mixup_fn：", mixup_fn)
    print("----------------------结束监视代码----------------------")
    exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(mixup_fn)： <class 'timm.data.mixup.Mixup'>
----------------------我的分割线1----------------------
mixup_fn： <timm.data.mixup.Mixup object at 0x7f5723411d50>
----------------------结束监视代码----------------------
```
这个`mixup_fn`的用法目前我先不深究了，以后用到的时候再详细研究。
最后，`def build_loader(config):`函数返回了`dataset_train`，`dataset_val`，`data_loader_train`，`data_loader_val`，`mixup_fn`这五个变量。我们最后再来看一下这五个变量都长什么样子。在`def build_loader(config):`函数定义的最后，试运行下述代码：
``` python
print("----------------------开始监视代码----------------------")
print("type(dataset_train)：", type(dataset_train))
print("----------------------我的分割线1----------------------")
print("dataset_train：", dataset_train)
print("----------------------我的分割线2----------------------")
print("type(dataset_val)：", type(dataset_val))
print("----------------------我的分割线3----------------------")
print("dataset_val：", dataset_val)
print("----------------------我的分割线4----------------------")
print("type(data_loader_train)：", type(data_loader_train))
print("----------------------我的分割线5----------------------")
print("data_loader_train：", data_loader_train)
print("----------------------我的分割线6----------------------")
print("type(data_loader_val)：", type(data_loader_val))
print("----------------------我的分割线7----------------------")
print("data_loader_val：", data_loader_val)
print("----------------------我的分割线8----------------------")
print("type(mixup_fn)：", type(mixup_fn))
print("----------------------我的分割线9----------------------")
print("mixup_fn：", mixup_fn)
print("----------------------结束监视代码----------------------")
exit()
return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
```
结果为：
```
----------------------开始监视代码----------------------
type(dataset_train)： <class 'torchvision.datasets.folder.ImageFolder'>
----------------------我的分割线1----------------------
dataset_train： Dataset ImageFolder
    Number of datapoints: 1281167
    Root location: /XXX-XXX/XXX_XXX_data/imagenet/origin/train
    StandardTransform
Transform: Compose(
               RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC)
               RandomXXXtalFlip(p=0.5)
               <timm.data.auto_augment.RandAugment object at 0x7f4beba11e90>
               ToTensor()
               Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
               <timm.data.random_erasing.RandomErasing object at 0x7f4beb44db10>
           )
----------------------我的分割线2----------------------
type(dataset_val)： <class 'torchvision.datasets.folder.ImageFolder'>
----------------------我的分割线3----------------------
dataset_val： Dataset ImageFolder
    Number of datapoints: 50000
    Root location: /XXX-XXX/XXX_XXX_data/imagenet/origin/val
    StandardTransform
Transform: Compose(
               Resize(size=256, interpolation=PIL.Image.BICUBIC)
               CenterCrop(size=(224, 224))
               ToTensor()
               Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
           )
----------------------我的分割线4----------------------
type(data_loader_train)： <class 'torch.utils.data.dataloader.DataLoader'>
----------------------我的分割线5----------------------
data_loader_train： <torch.utils.data.dataloader.DataLoader object at 0x7f4ba77d3c10>
----------------------我的分割线6----------------------
type(data_loader_val)： <class 'torch.utils.data.dataloader.DataLoader'>
----------------------我的分割线7----------------------
data_loader_val： <torch.utils.data.dataloader.DataLoader object at 0x7f4ba77d3c90>
----------------------我的分割线8----------------------
type(mixup_fn)： <class 'timm.data.mixup.Mixup'>
----------------------我的分割线9----------------------
mixup_fn： <timm.data.mixup.Mixup object at 0x7f4ba77d3d10>
----------------------结束监视代码----------------------
```
至此，`def build_loader(config):`这个函数分析完毕。`Swin Transformer`的训练代码继续回到主函数`main(config)`里面。我们前边所做的这如此之长的分析，仅仅分析完了主函数的这一句代码：
``` python
def main(config):
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config)
```
到目前为止，主函数刚刚初始化了训练数据集`dataset_train`、验证数据集`dataset_val`、训练数据加载器`data_loader_train`、验证数据加载器`data_loader_val`以及`mixup_fn`这五个变量（其中`mixup_fn`变量的具体功能我暂时不知道）。这五个变量在接下来的代码中将会用到，下面会详细地分析这五个变量都是怎么被用上的。

接下来的代码会输出一行日志：
``` python
logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
```
我们来测试一下，看看这行日志输出了什么。试运行下述代码：
``` python
print("----------------------开始监视代码----------------------")
logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
[2021-12-01 15:55:22 swin_tiny_patch4_window7_224](main.py 142): INFO Creating model:swin/swin_tiny_patch4_window7_224
----------------------结束监视代码----------------------
```
这行代码其实就是以日志的格式输出了一下我们要构造的模型是哪个模型。对Swin Transformer模型的具体分析，以后有时间再补全。

接下来的三行代码就是利用已知参数构造了模型，并把模型调整到GPU上，并把模型的全貌打印出来：
``` python
model = build_model(config)
model.cuda()
logger.info(str(model))
```
我们来看看我们的模型长什么样子。试运行如下的代码：
``` python
model = build_model(config)
print("----------------------开始监视代码----------------------")
print("type(model)：", type(model))
print("----------------------我的分割线1----------------------")
print("model：", model)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(model)： <class 'models.swin_transformer.SwinTransformer'>
----------------------我的分割线1----------------------
model： SwinTransformer(
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
至此，我们所使用的Swin Transformer模型，终于展现出了它的全貌了。我目前暂且不需要深究这个模型的细节，以后有需要，再研究模型细节。对于`model.cuda()`这行代码，我们利用vscode的定义跳转功能进入到`model.cuda()`函数的定义中，可以看到，`model.cuda()`函数的功能是：将所有的模型参数和缓冲区都移动到GPU上。缓冲区的具体含义我也不清楚，以后如果需要再补全。至于`logger.info(str(model))`这行代码，它的功能就是把模型的全貌输出成一个字符串。我们再来测试一下下述代码，输出一下模型结构：
``` python
print("----------------------开始监视代码----------------------")
logger.info(str(model))
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
[2021-12-02 16:29:39 swin_tiny_patch4_window7_224](main.py 154): INFO SwinTransformer(
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
因此，`logger.info(str(model))`这行代码其实就是让我们看一下我们所用的模型长什么样子而已，没有什么特别新奇的内容。

接下来的一行代码构造了优化器：
``` python
optimizer = build_optimizer(config, model)
```
这一行代码的输入`config`和`model`，由于我们之前已经看过这两个变量的模样了，在此不再赘述。我们就直接看一下`optimizer`的样子。测试如下代码：
``` python
optimizer = build_optimizer(config, model)
print("----------------------开始监视代码----------------------")
print("type(optimizer)：", type(optimizer))
print("----------------------我的分割线1----------------------")
print("optimizer：", optimizer)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(optimizer)： <class 'torch.optim.adamw.AdamW'>
----------------------我的分割线1----------------------
optimizer： AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 3.125e-05
    weight_decay: 0.05

Parameter Group 1
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 3.125e-05
    weight_decay: 0.0
)
----------------------结束监视代码----------------------
```
构造优化器的核心是调用了`build_optimizer(config, model)`函数。这个`build_optimizer(config, model)`函数位于`/Swin-Transformer/optimizer.py`脚本里。这份脚本里有构造优化器和设定权衰减相关的函数。以后如果要构造新的优化器或者修改权衰减相关的代码，就要修改`/Swin-Transformer/optimizer.py`脚本里的相应代码。

接下来的一行代码是利用`apex`这个混合精度训练工具对模型和优化器进行重构：
``` python
if config.AMP_OPT_LEVEL != "O0":
    model, optimizer = amp.initialize(
        model, optimizer, opt_level=config.AMP_OPT_LEVEL
    )
```
关于`apex`这个混合精度训练工具，根据我粗略的了解，可以极大地加快训练速度。以后有机会可以学学。但目前我暂且先略过。上面一行代码用到的核心函数`amp.initialize()`位于`/data/XXX/conda_env/swin/lib/python3.7/site-packages/apex/amp/frontend.py`里面。经过我的测试，直接把经`apex`重构后的模型和优化器print出来，也看不出任何区别。如果以后要深入学习`apex`混合精度训练的用法，可以参考[这篇文章](https://zhuanlan.zhihu.com/p/79887894)。

接下来的一行代码对模型进行了适合于分布式训练的处理：
``` python
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
)
```
PyTorch官方的`torch.nn.parallel.DistributedDataParallel()`函数的作用是：在模型层面，实现了基于`torch.distributed`模块的数据并行。更详细的信息参考[PyTorch官方torch.nn.parallel.DistributedDataParallel文档](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。

接下来是这样的一行代码：
``` python
model_without_ddp = model.module
```
理解这行代码的关键在于搞清楚：`model.module`究竟是什么东西？我们来看一下`model`和`model.module`都长什么样子。测试下述代码：
``` python
print("----------------------开始监视代码----------------------")
print("type(model)：", type(model))
print("----------------------我的分割线1----------------------")
print("model：", model)
print("----------------------我的分割线2----------------------")
print("type(model.module)：", type(model.module))
print("----------------------我的分割线3----------------------")
print("model.module：", model.module)
print("----------------------结束监视代码----------------------")
exit()
model_without_ddp = model.module
```
结果为：
```
----------------------开始监视代码----------------------
type(model)： <class 'torch.nn.parallel.distributed.DistributedDataParallel'>
----------------------我的分割线1----------------------
model： DistributedDataParallel(
  (module): SwinTransformer(
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
)
----------------------我的分割线2----------------------
type(model.module)： <class 'models.swin_transformer.SwinTransformer'>
----------------------我的分割线3----------------------
model.module： SwinTransformer(
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
由此就明白了，`model`变量的类型是：`<class 'torch.nn.parallel.distributed.DistributedDataParallel'>`。`model_without_ddp = model.module`这句话，就是把没有分布式数据并行DistributedDataParallel的模型保存在`model_without_ddp`变量里。因此，`model`变量保存的是DistributedDataParallel分布式数据并行化的模型，`model_without_ddp`变量里保存的是原始的`<class 'models.swin_transformer.SwinTransformer'>`模型。因此，**一个`<class 'torch.nn.parallel.distributed.DistributedDataParallel'>`类型的模型`model`调用`model.module`得到的就是一个原始的没有分布式数据并行化的模型。**

接下来的两行代码输出了参数的个数：
``` python
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"number of params: {n_parameters}")
```
从这两行代码里，我们需要学会的就是，如何在我们未来的模型里获得参数个数。我们来测试一下下述代码：
``` python
print("----------------------开始监视代码----------------------")
print("type(model)：", type(model))
print("----------------------我的分割线1----------------------")
print("model：", model)
print("----------------------我的分割线2----------------------")
print("type(model.parameters())：", type(model.parameters()))
print("----------------------我的分割线3----------------------")
print("model.parameters()：", model.parameters())
print("----------------------结束监视代码----------------------")
exit()
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"number of params: {n_parameters}")
```
结果为：
```
----------------------开始监视代码----------------------
type(model)： <class 'torch.nn.parallel.distributed.DistributedDataParallel'>
----------------------我的分割线1----------------------
model： DistributedDataParallel(
  (module): SwinTransformer(
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
)
----------------------我的分割线2----------------------
type(model.parameters())： <class 'generator'>
----------------------我的分割线3----------------------
model.parameters()： <generator object Module.parameters at 0x7f131c3807d0>
----------------------结束监视代码----------------------
```
由此我们就明白了：一个`<class 'torch.nn.parallel.distributed.DistributedDataParallel'>`类型的模型`model`调用了它的`model.parameters()`函数，得到的就是一个生成器。（[参见`model.parameters()`的PyTorch官方文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters)）如果使用`for`循环语句遍历这个生成器，得到的就是模型`model`的所有参数。`p.numel() for p in model.parameters() if p.requires_grad`这句话中的`p.requires_grad`是一个逻辑值，表明一个参数是不是可训练的参数。我们再来详细地看看这些可学习参数都是什么类型和形状。测试下述代码：
``` python
print("----------------------开始监视代码----------------------")
for p in model.parameters():
    if p.requires_grad:
        print(f"类型为{type(p)}形状为{p.shape}的可学习参数p的p.numel()的值是 {p.numel()}")
print("----------------------结束监视代码----------------------")
exit()
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"number of params: {n_parameters}")
```
结果为：
```
----------------------开始监视代码----------------------
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96, 3, 4, 4])的可学习参数p的p.numel()的值是 4608
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 3])的可学习参数p的p.numel()的值是 507
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([288, 96])的可学习参数p的p.numel()的值是 27648
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([288])的可学习参数p的p.numel()的值是 288
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96, 96])的可学习参数p的p.numel()的值是 9216
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 96])的可学习参数p的p.numel()的值是 36864
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96, 384])的可学习参数p的p.numel()的值是 36864
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 3])的可学习参数p的p.numel()的值是 507
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([288, 96])的可学习参数p的p.numel()的值是 27648
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([288])的可学习参数p的p.numel()的值是 288
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96, 96])的可学习参数p的p.numel()的值是 9216
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 96])的可学习参数p的p.numel()的值是 36864
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96, 384])的可学习参数p的p.numel()的值是 36864
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([96])的可学习参数p的p.numel()的值是 96
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192, 384])的可学习参数p的p.numel()的值是 73728
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 6])的可学习参数p的p.numel()的值是 1014
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([576, 192])的可学习参数p的p.numel()的值是 110592
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([576])的可学习参数p的p.numel()的值是 576
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192, 192])的可学习参数p的p.numel()的值是 36864
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768, 192])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192, 768])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 6])的可学习参数p的p.numel()的值是 1014
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([576, 192])的可学习参数p的p.numel()的值是 110592
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([576])的可学习参数p的p.numel()的值是 576
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192, 192])的可学习参数p的p.numel()的值是 36864
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768, 192])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192, 768])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([192])的可学习参数p的p.numel()的值是 192
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 768])的可学习参数p的p.numel()的值是 294912
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 12])的可学习参数p的p.numel()的值是 2028
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152, 384])的可学习参数p的p.numel()的值是 442368
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152])的可学习参数p的p.numel()的值是 1152
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 384])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536, 384])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536])的可学习参数p的p.numel()的值是 1536
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 1536])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 12])的可学习参数p的p.numel()的值是 2028
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152, 384])的可学习参数p的p.numel()的值是 442368
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152])的可学习参数p的p.numel()的值是 1152
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 384])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536, 384])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536])的可学习参数p的p.numel()的值是 1536
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 1536])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 12])的可学习参数p的p.numel()的值是 2028
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152, 384])的可学习参数p的p.numel()的值是 442368
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152])的可学习参数p的p.numel()的值是 1152
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 384])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536, 384])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536])的可学习参数p的p.numel()的值是 1536
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 1536])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 12])的可学习参数p的p.numel()的值是 2028
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152, 384])的可学习参数p的p.numel()的值是 442368
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152])的可学习参数p的p.numel()的值是 1152
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 384])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536, 384])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536])的可学习参数p的p.numel()的值是 1536
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 1536])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 12])的可学习参数p的p.numel()的值是 2028
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152, 384])的可学习参数p的p.numel()的值是 442368
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152])的可学习参数p的p.numel()的值是 1152
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 384])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536, 384])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536])的可学习参数p的p.numel()的值是 1536
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 1536])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 12])的可学习参数p的p.numel()的值是 2028
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152, 384])的可学习参数p的p.numel()的值是 442368
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1152])的可学习参数p的p.numel()的值是 1152
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 384])的可学习参数p的p.numel()的值是 147456
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536, 384])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536])的可学习参数p的p.numel()的值是 1536
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384, 1536])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([384])的可学习参数p的p.numel()的值是 384
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768, 1536])的可学习参数p的p.numel()的值是 1179648
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536])的可学习参数p的p.numel()的值是 1536
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1536])的可学习参数p的p.numel()的值是 1536
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 24])的可学习参数p的p.numel()的值是 4056
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([2304, 768])的可学习参数p的p.numel()的值是 1769472
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([2304])的可学习参数p的p.numel()的值是 2304
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768, 768])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([3072, 768])的可学习参数p的p.numel()的值是 2359296
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([3072])的可学习参数p的p.numel()的值是 3072
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768, 3072])的可学习参数p的p.numel()的值是 2359296
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([169, 24])的可学习参数p的p.numel()的值是 4056
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([2304, 768])的可学习参数p的p.numel()的值是 1769472
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([2304])的可学习参数p的p.numel()的值是 2304
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768, 768])的可学习参数p的p.numel()的值是 589824
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([3072, 768])的可学习参数p的p.numel()的值是 2359296
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([3072])的可学习参数p的p.numel()的值是 3072
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768, 3072])的可学习参数p的p.numel()的值是 2359296
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([768])的可学习参数p的p.numel()的值是 768
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1000, 768])的可学习参数p的p.numel()的值是 768000
类型为<class 'torch.nn.parameter.Parameter'>形状为torch.Size([1000])的可学习参数p的p.numel()的值是 1000
----------------------结束监视代码----------------------
```
由此就明白了：PyTorch构造的模型参数的变量类型通常来说是`<class 'torch.nn.parameter.Parameter'>`类型的变量。对`<class 'torch.nn.parameter.Parameter'>`类型的模型参数变量`p`调用`p.numel()`方法，得到的就是这个参数的个数。这个参数的个数其实就是模型参数变量`p`的张量维数的乘积。

接下来是一个if条件语句：
``` python
if hasattr(model_without_ddp, "flops"):
    flops = model_without_ddp.flops()
    logger.info(f"number of GFLOPs: {flops / 1e9}")
```
`hasattr`函数是Python的一个内置函数，参考[Python hasattr官方文档](https://docs.python.org/3/library/functions.html#hasattr)，这个`hasattr`函数的作用是：判断某个字符串是否是这个对象的属性。我们在空白脚本中测试如下的代码：
``` python
# 以下代码在空白脚本中运行
class myclass:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def myfunction(self, alpha):
        self.alpha = alpha


myinstance = myclass(1, 2, 3)
print('hasattr(myinstance, "x")：', hasattr(myinstance, "x"))
print('hasattr(myinstance, "y")：', hasattr(myinstance, "y"))
print('hasattr(myinstance, "z")：', hasattr(myinstance, "z"))
print('hasattr(myinstance, "alpha")：', hasattr(myinstance, "alpha"))

myinstance.myfunction(10)
print('hasattr(myinstance, "alpha")：', hasattr(myinstance, "alpha"))
```
结果为：
```
hasattr(myinstance, "x")： True
hasattr(myinstance, "y")： True
hasattr(myinstance, "z")： True
hasattr(myinstance, "alpha")： False
hasattr(myinstance, "alpha")： True
```
再来测试一下对类的方法`hasattr`函数会返回什么。在空白脚本中测试如下代码：
``` python
# 以下代码在空白脚本中运行
class myclass:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def myfunction(self, alpha, beta):
        self.alpha = alpha
        beta_shuxing = beta


myinstance = myclass(1, 2, 3)
print('hasattr(myinstance, "myfunction")：', hasattr(myinstance, "myfunction"))
```
结果为：
```
hasattr(myinstance, "myfunction")： True
```
由此就明白了：Python的内置函数`hasattr`函数的作用是：判断某个字符串是否是这个对象的属性。对象通常是某个类的实例，而这个对象的属性就是这个类在`__init__`方法中所定义的诸如`self.x`，`self.y`之类的属性或者是这个类的某个方法所定义的诸如上面演示中`self.alpha`这样的属性。某个属性`self.x`的名称必须精确为`x`，而不能是大写的`X`。Python的内置函数`hasattr`函数对于类的方法也可以返回`True`。
我们来整体运行一下这个`if`条件语句。测试如下的代码：
``` python
print("----------------------开始监视代码----------------------")
if hasattr(model_without_ddp, "flops"):
    flops = model_without_ddp.flops()
    logger.info(f"number of GFLOPs: {flops / 1e9}")
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
[2021-12-13 19:58:40 swin_tiny_patch4_window7_224](main.py 215): INFO number of GFLOPs: 4.49440512
----------------------结束监视代码----------------------
```
`GFLOPs`这个指标的含义是`每秒浮点运算次数(FLOPS)`，它是一个训练速度的指标。从上面的代码里，我们可以学到：用模型的`.flops()`函数来得到模型的每秒浮点运算次数（也就是代码中的`model_without_ddp.flops()`这句话）。

接下来是构造学习率调度器的代码：
``` python
lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
```
这行代码就是利用`build_scheduler()`函数来构造学习率调度器。`build_scheduler()`函数的完整代码如下（在`/Swin-Transformer/lr_scheduler.py`脚本里）：
``` python
def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler
```
关于这个学习率衰减的具体细节，我暂且先略过了。以后如果需要自己设计学习率衰减模块，或许会再来深入地学习这些代码的细节这里，我们只需看一下学习率衰减模块的类型和样子就好。测试如下的代码：
``` python
lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
print("----------------------开始监视代码----------------------")
print("type(lr_scheduler)：", type(lr_scheduler))
print("----------------------我的分割线1----------------------")
print("lr_scheduler：", lr_scheduler)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(lr_scheduler)： <class 'timm.scheduler.cosine_lr.CosineLRScheduler'>
----------------------我的分割线1----------------------
lr_scheduler： <timm.scheduler.cosine_lr.CosineLRScheduler object at 0x7fd6340e5b50>
----------------------结束监视代码----------------------
```
由此知，学习率衰减器的构建是利用`timm`这个库来构建的。本次Swin Transformer的训练使用了余弦学习率衰减方案。

接下来的代码是一个if条件语句：
``` python
if config.AUG.MIXUP > 0.0:
    # smoothing is handled with mixup label transform
    criterion = SoftTargetCrossEntropy()
elif config.MODEL.LABEL_SMOOTHING > 0.0:
    criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
else:
    criterion = torch.nn.CrossEntropyLoss()

max_accuracy = 0.0
```
这段代码主要的目的就是：按照不同的设定条件，构造`criterion`变量。我们来看看这个`criterion`变量是什么东西。测试如下的代码：
``` python
if config.AUG.MIXUP > 0.0:
    # smoothing is handled with mixup label transform
    criterion = SoftTargetCrossEntropy()
elif config.MODEL.LABEL_SMOOTHING > 0.0:
    criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
else:
    criterion = torch.nn.CrossEntropyLoss()
print("----------------------开始监视代码----------------------")
print("type(criterion)：", type(criterion))
print("----------------------我的分割线1----------------------")
print("criterion：", criterion)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(criterion)： <class 'timm.loss.cross_entropy.SoftTargetCrossEntropy'>
----------------------我的分割线1----------------------
criterion： SoftTargetCrossEntropy()
----------------------结束监视代码----------------------
```
由此可见，这个`criterion`变量是一个`<class 'timm.loss.cross_entropy.SoftTargetCrossEntropy'>`类型的变量。由此大概可以推测到，这个`criterion`变量应该是和Swin Transformer的Loss函数有关。以后当用到这个`criterion`变量的时候，再来详细地分析。


接下来的代码是三个`if`条件语句：
``` python
if config.TRAIN.AUTO_RESUME:
    resume_file = auto_resume_helper(config.OUTPUT)
    if resume_file:
        if config.MODEL.RESUME:
            logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
        config.defrost()
        config.MODEL.RESUME = resume_file
        config.freeze()
        logger.info(f'auto resuming from {resume_file}')
    else:
        logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

if config.MODEL.RESUME:
    max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
    acc1, acc5, loss = validate(config, data_loader_val, model)
    logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
    if config.EVAL_MODE:
        return

if config.THROUGHPUT_MODE:
    throughput(data_loader_val, model, logger)
    return
```
参考`/Swin-Transformer/config.py`脚本里给出的注释，`if config.TRAIN.AUTO_RESUME:`条件语句是在判断是否从最新的检测点处恢复训练；`if config.MODEL.RESUME:`条件语句是在判断是否给定了所要从其处恢复训练的模型；`if config.THROUGHPUT_MODE:`条件语句是在判断是否启动仅仅测试吞吐量（虽然我也不太明白这个究竟是什么意思）。这三个条件判断语句的细节，我暂且不深究了，之后有时间，再来看细节。

主函数里，剩下的所有代码就是下面的训练代码了：
``` python
logger.info("Start training")
start_time = time.time()
for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    data_loader_train.sampler.set_epoch(epoch)

    train_one_epoch(
        config,
        model,
        criterion,
        data_loader_train,
        optimizer,
        epoch,
        mixup_fn,
        lr_scheduler,
    )
    if dist.get_rank() == 0 and (
        epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
    ):
        save_checkpoint(
            config,
            epoch,
            model_without_ddp,
            max_accuracy,
            optimizer,
            lr_scheduler,
            logger,
        )

    acc1, acc5, loss = validate(config, data_loader_val, model)
    logger.info(
        f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
    )
    max_accuracy = max(max_accuracy, acc1)
    logger.info(f"Max accuracy: {max_accuracy:.2f}%")

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
logger.info("Training time {}".format(total_time_str))
```
我们来一行一行地分析这里的训练代码。首先是下面的两行：
``` python
logger.info("Start training")
start_time = time.time()
```
这两行代码的功能是：通过日志输出开始训练的标记（在这里是一个字符串`Start training`），并记录训练开始的时间。从这里我们可以看到：以后如果我要监测训练的时间，也应该像Swin Transformer的作者这样，在开始迭代epoch的for循环之前加入代码`start_time = time.time()`（其中`time`模块是Python自带的标准库）。

接下来正式进入了训练循环：
``` python
for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    data_loader_train.sampler.set_epoch(epoch)
```
`data_loader_train.sampler.set_epoch(epoch)`这句话是和分布式训练有关的。在[PyTorch官方torch.utils.data.distributed.DistributedSampler文档](https://pytorch.org/docs/1.8.0/data.html#torch.utils.data.distributed.DistributedSampler)里，解释了这样做的原因。

下面的一行代码是训练一个完整的epoch的代码：
``` python
train_one_epoch(
    config,
    model,
    criterion,
    data_loader_train,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
)
```
这行代码是训练最最核心的代码，需要仔细研究。我们来看看这行代码的详细内容。

### 训练一个epoch的函数train_one_epoch()
下面是`train_one_epoch()`函数的完整代码：
``` python
def train_one_epoch(
    config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD
                    )
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
```
在训练一个epoch的函数`train_one_epoch()`里，首先是将模型调整为训练模式，并且清零梯度：
``` python
model.train()
optimizer.zero_grad()
```
这两行代码是很常规的代码，它们的用法必须记住。

接下来是这样的四行代码：
``` python
num_steps = len(data_loader)
batch_time = AverageMeter()
loss_meter = AverageMeter()
norm_meter = AverageMeter()
```
`num_steps = len(data_loader)`这行代码记录了步骤数。下面三行代码都是初始化了`AverageMeter`类的实例。这个`AverageMeter`类是在`/conda_env/swin/lib/python3.7/site-packages/timm/utils/metrics.py`这个脚本里，属于`timm`库的一部分。

接下来正式开始了训练的过程：
``` python
start = time.time()
end = time.time()
for idx, (samples, targets) in enumerate(data_loader):
    samples = samples.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
```
我们来看看对训练数据加载器`data_loader`进行枚举，得到的是什么。测试如下的代码：
``` python
for idx, (samples, targets) in enumerate(data_loader):
    print("----------------------开始监视代码----------------------")
    print("type(samples)：", type(samples))
    print("----------------------我的分割线1----------------------")
    print("samples.shape：", samples.shape)
    print("----------------------我的分割线2----------------------")
    print("type(targets)：", type(targets))
    print("----------------------我的分割线3----------------------")
    print("targets.shape：", targets.shape)
    print("----------------------结束监视代码----------------------")
    exit()
    samples = samples.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
```
结果为：
```
----------------------开始监视代码----------------------
type(samples)： <class 'torch.Tensor'>
----------------------我的分割线1----------------------
samples.shape： torch.Size([32, 3, 224, 224])
----------------------我的分割线2----------------------
type(targets)： <class 'torch.Tensor'>
----------------------我的分割线3----------------------
targets.shape： torch.Size([32])
----------------------结束监视代码----------------------
```
由此就明白了：使用`for idx, (samples, targets) in enumerate(data_loader):`语句对训练用dataloader进行枚举的时候，得到的`samples`和`targets`对象都是PyTorch张量。`samples`代表的是数据样本，而`targets`代表的是类别标签。`samples = samples.cuda(non_blocking=True)`和`targets = targets.cuda(non_blocking=True)`这两行代码是在把数据移动到GPU上（参考[torch.Tensor.cuda官方文档](https://pytorch.org/docs/1.8.0/tensors.html#torch.Tensor.cuda)）。至于为什么要设定`non_blocking=True`，同样参考[torch.Tensor.cuda官方文档](https://pytorch.org/docs/1.8.0/tensors.html#torch.Tensor.cuda)或者参考[这里的解释](https://blog.csdn.net/qq_37297763/article/details/116670668)。

接下来的两行代码对训练数据和标签做了一个变换：
``` python
if mixup_fn is not None:
    samples, targets = mixup_fn(samples, targets)
```
这两行代码的核心是`mixup_fn`函数。我们首先来看看这个`mixup_fn`函数究竟是什么。测试如下的代码：
``` python
if mixup_fn is not None:
    print("----------------------开始监视代码----------------------")
    print("type(mixup_fn)：", type(mixup_fn))
    print("----------------------我的分割线1----------------------")
    print("mixup_fn：", mixup_fn)
    print("----------------------结束监视代码----------------------")
    exit()
    samples, targets = mixup_fn(samples, targets)
```
结果为：
```
----------------------开始监视代码----------------------
type(mixup_fn)： <class 'timm.data.mixup.Mixup'>
----------------------我的分割线1----------------------
mixup_fn： <timm.data.mixup.Mixup object at 0x7f03571163d0>
----------------------结束监视代码----------------------
```
由此知，这个`mixup_fn`函数是一个`<class 'timm.data.mixup.Mixup'>`类型的对象，它是`timm`库的一部分。它的目的是：对数据进行增强。然后，我们再来看看数据经过`mixup_fn`函数映射后发生了什么变换。测试如下的代码：
``` python
if mixup_fn is not None:
    samples, targets = mixup_fn(samples, targets)
    print("----------------------开始监视代码----------------------")
    print("type(samples)：", type(samples))
    print("----------------------我的分割线1----------------------")
    print("samples.shape：", samples.shape)
    print("----------------------我的分割线2----------------------")
    print("type(targets)：", type(targets))
    print("----------------------我的分割线3----------------------")
    print("targets.shape：", targets.shape)
    print("----------------------结束监视代码----------------------")
    exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(samples)： <class 'torch.Tensor'>
----------------------我的分割线1----------------------
samples.shape： torch.Size([32, 3, 224, 224])
----------------------我的分割线2----------------------
type(targets)： <class 'torch.Tensor'>
----------------------我的分割线3----------------------
targets.shape： torch.Size([32, 1000])
----------------------结束监视代码----------------------
```
从这里可以看到，`targets.shape`发生了变换。在`mixup_fn()`函数映射之前，`target`变量是一个32维的张量。这个32维张量里存储的是每张图片的类别序号。但是经过`mixup_fn()`函数映射之后，这个每个类别序号变成了一个1000维的张量。这个1000维的张量是一个混合（mixup）标签。经过查阅相关资料（比如[这里](https://www.zhihu.com/question/308572298)）可知，`mixup_fn`函数的功能是：对数据进行增强。这种数据增强技巧主要用于图像分类任务。更多的细节可以参考这两篇论文：[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412v2)，[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187v2)。

接下来的一行代码就是应用模型进行推理的核心代码了：
``` python
outputs = model(samples)
```
我们先来看看模型的输入和输出。试运行下述代码：
``` python
outputs = model(samples)
print("----------------------开始监视代码----------------------")
print("type(samples)：", type(samples))
print("----------------------我的分割线1----------------------")
print("samples.shape：", samples.shape)
print("----------------------我的分割线2----------------------")
print("type(outputs)：", type(outputs))
print("----------------------我的分割线3----------------------")
print("outputs.shape：", outputs.shape)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(samples)： <class 'torch.Tensor'>
----------------------我的分割线1----------------------
samples.shape： torch.Size([32, 3, 224, 224])
----------------------我的分割线2----------------------
type(outputs)： <class 'torch.Tensor'>
----------------------我的分割线3----------------------
outputs.shape： torch.Size([32, 1000])
----------------------结束监视代码----------------------
```
由此就很清楚了：Swin Transformer模型接收的输入，是32（32是我设定的batch size）张224x224的RGB图像，输出的是32个1000维的张量。每个张量给出了一个类别概率。我们再来重新看一下模型的样子。测试如下的代码：
``` python
outputs = model(samples)
print("----------------------开始监视代码----------------------")
print("type(model)：", type(model))
print("----------------------我的分割线1----------------------")
print("model：", model)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(model)： <class 'torch.nn.parallel.distributed.DistributedDataParallel'>
----------------------我的分割线1----------------------
model： DistributedDataParallel(
  (module): SwinTransformer(
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
)
----------------------结束监视代码----------------------
```
可以看到，我们使用的是`<class 'torch.nn.parallel.distributed.DistributedDataParallel'>`类型的多卡模型来做的推理。

接下来就是关于loss的代码了：
``` python
if config.TRAIN.ACCUMULATION_STEPS > 1:
    loss = criterion(outputs, targets)
    loss = loss / config.TRAIN.ACCUMULATION_STEPS
    if config.AMP_OPT_LEVEL != "O0":
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
            )
        else:
            grad_norm = get_grad_norm(amp.master_params(optimizer))
    else:
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD
            )
        else:
            grad_norm = get_grad_norm(model.parameters())
    if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step_update(epoch * num_steps + idx)
else:
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    if config.AMP_OPT_LEVEL != "O0":
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
            )
        else:
            grad_norm = get_grad_norm(amp.master_params(optimizer))
    else:
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD
            )
        else:
            grad_norm = get_grad_norm(model.parameters())
    optimizer.step()
    lr_scheduler.step_update(epoch * num_steps + idx)
```
经测试，这段代码会执行`else`语句的部分。我们来仔细看一下`else`语句的部分：
``` python
loss = criterion(outputs, targets)
optimizer.zero_grad()
if config.AMP_OPT_LEVEL != "O0":
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    if config.TRAIN.CLIP_GRAD:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
        )
    else:
        grad_norm = get_grad_norm(amp.master_params(optimizer))
else:
    loss.backward()
    if config.TRAIN.CLIP_GRAD:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.TRAIN.CLIP_GRAD
        )
    else:
        grad_norm = get_grad_norm(model.parameters())
optimizer.step()
lr_scheduler.step_update(epoch * num_steps + idx)
```
首先，第一行代码`loss = criterion(outputs, targets)`的作用是计算loss。前面我们已经推测过了，这个`criterion`变量是一个`<class 'timm.loss.cross_entropy.SoftTargetCrossEntropy'>`类型的变量。由此大概可以推测到，这个`criterion`变量应该是和Swin Transformer的loss函数有关。至此，印证了我们之前的推测。我们来看看要计算loss需要的输入都长什么样。测试下述代码（注意，这里用到一个PyTorch `.size()`和`.shape`的区别。其实它俩没有区别。参见[Github上PyTorch作者的解释](https://github.com/pytorch/pytorch/issues/5544)）：
``` python
loss = criterion(outputs, targets)
print("----------------------开始监视代码----------------------")
print("type(outputs)：", type(outputs))
print("----------------------我的分割线1----------------------")
print("outputs.shape：", outputs.shape)
print("----------------------我的分割线2----------------------")
print("type(targets)：", type(targets))
print("----------------------我的分割线3----------------------")
print("targets.shape：", targets.shape)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(outputs)： <class 'torch.Tensor'>
----------------------我的分割线1----------------------
outputs.shape： torch.Size([32, 1000])
----------------------我的分割线2----------------------
type(targets)： <class 'torch.Tensor'>
----------------------我的分割线3----------------------
targets.shape： torch.Size([32, 1000])
----------------------结束监视代码----------------------
```
由此就明白了：计算loss的时候，是在把实际网络推理出来的类别概率向量（在这里，是1000维的向量）和标签类别向量（在这里，也是1000维的向量。只不过是一个经过mixup混合后的标签向量。）
再来看看计算出的loss长什么样。测试下述代码：
``` python
loss = criterion(outputs, targets)
print("----------------------开始监视代码----------------------")
print("type(loss)：", type(loss))
print("----------------------我的分割线1----------------------")
print("loss：", loss)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
type(loss)： <class 'torch.Tensor'>
----------------------我的分割线1----------------------
loss： tensor(6.9016, device='cuda:0', grad_fn=<MeanBackward0>)
----------------------结束监视代码----------------------
```
所以我们知道了：**PyTorch神经网络输出的loss，并不是一个浮点数，而是一个1维的PyTorch张量，并且这个loss张量还是带有设备标识和梯度函数的。**

下面的一行`optimizer.zero_grad()`是优化器梯度清零，这个是标准步骤，没有什么可说的。

接下来的代码与梯度的反向传播有关：
``` python
if config.AMP_OPT_LEVEL != "O0":
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    if config.TRAIN.CLIP_GRAD:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
        )
    else:
        grad_norm = get_grad_norm(amp.master_params(optimizer))
else:
    loss.backward()
    if config.TRAIN.CLIP_GRAD:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.TRAIN.CLIP_GRAD
        )
    else:
        grad_norm = get_grad_norm(model.parameters())
```
经测试，代码会执行`if config.AMP_OPT_LEVEL != "O0":`的部分。因此我们来详细地看一下`if config.AMP_OPT_LEVEL != "O0":`的部分代码：
``` python
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
if config.TRAIN.CLIP_GRAD:
    grad_norm = torch.nn.utils.clip_grad_norm_(
        amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
    )
else:
    grad_norm = get_grad_norm(amp.master_params(optimizer))
```
这段代码中的前两行：
``` python
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```
是使用`apex`混合精度加速工具进行梯度反向传播的代码。这个用法是固定的，只需要记住就行了。以后可以再查查相关的文档。

接下来将会执行的是下面的`if config.TRAIN.CLIP_GRAD:`语句包裹的内容：
``` python
grad_norm = torch.nn.utils.clip_grad_norm_(
    amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
)
```
关于这行代码的核心`torch.nn.utils.clip_grad_norm_()`函数，参考[PyTorch官方TORCH.NN.UTILS.CLIP_GRAD_NORM_文档](https://pytorch.org/docs/1.7.1/generated/torch.nn.utils.clip_grad_norm_.html#torch-nn-utils-clip-grad-norm)。简单来说，`torch.nn.utils.clip_grad_norm_()`函数实现的功能是：计算梯度的范数。我们来看看计算出来的梯度范数是什么。测试如下的代码：
``` python
grad_norm = torch.nn.utils.clip_grad_norm_(
    amp.master_params(optimizer), config.TRAIN.CLIP_GRAD
)
print("----------------------开始监视代码----------------------")
print("grad_norm：", grad_norm)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
grad_norm： tensor(7.7471, device='cuda:0')
----------------------结束监视代码----------------------
```
可以看到，经过`torch.nn.utils.clip_grad_norm_()`函数映射后得到的梯度范数，仍然是一个一维的梯度张量，但是没有了梯度函数，只有梯度的值和设备。

下面的两行代码是执行优化和调整学习率的代码：
``` python
optimizer.step()
lr_scheduler.step_update(epoch * num_steps + idx)
```
`optimizer.step()`这句话的作用参见[PyTorch官方torch.optim.Optimizer.step文档](https://pytorch.org/docs/1.7.1/optim.html#torch.optim.Optimizer.step)，这句话是为了执行参数更新的步骤。而`lr_scheduler.step_update(epoch * num_steps + idx)`这句话则是调整学习率，对学习率进行衰减。

下一行是这样的一行代码：
``` python
torch.cuda.synchronize()
```
这行代码的PyTorch官方文档，参见[这里](https://pytorch.org/docs/1.7.1/cuda.html#torch.cuda.synchronize)。它的作用是：等待CUDA设备上所有流中的所有内核都完成。具体的含义我也不是很清楚，之后再查查。

接下来是这样的四行代码：
``` python
loss_meter.update(loss.item(), targets.size(0))
norm_meter.update(grad_norm)
batch_time.update(time.time() - end)
end = time.time()
```
这四行代码的前三行，都是调用`/conda_env/swin/lib/python3.7/site-packages/timm/utils/metrics.py`中`class AverageMeter:`类的`def update(self, val, n=1):`函数。这个函数的核心作用就是计算输入的某个平均值。

接下来的代码就是打印出一个关于训练状况的字符串：
``` python
if idx % config.PRINT_FREQ == 0:
    lr = optimizer.param_groups[0]["lr"]
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    etas = batch_time.avg * (num_steps - idx)
    logger.info(
        f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
        f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
        f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
        f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
        f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
        f"mem {memory_used:.0f}MB"
    )
```
这段代码，就是打印出一个监视训练状况的字符串，没有什么特别之处，只需要记住就行了。在这段代码之下还有一个记录整个epoch训练时间的代码，由于就是固定的用法，所以我先不在这个笔记中放上来了。
 
至此，Swin Transformer单个epoch的训练完成了。

### 训练完一个epoch以后，继续回到训练代码
训练完一个epoch之后，首先做的是在主进程中保存checkpoint：
``` python
if dist.get_rank() == 0 and (
    epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
):
    save_checkpoint(
        config,
        epoch,
        model_without_ddp,
        max_accuracy,
        optimizer,
        lr_scheduler,
        logger,
    )
```
这段代码本身没有什么特别之处，主要的内容是在`save_checkpoint()`这个函数里完成的。对这个函数的实现细节，我暂且先不深究，只需知道：**保存训练中途的模型以及训练完的模型，是在/Swin-Transformer/utils.py里的save_checkpoint()函数里完成的**。如果我要进行保存模型方面的修改，就应该进入/Swin-Transformer/utils.py里的save_checkpoint()函数里进行相应的修改。由此可见，Swin Transformer的代码，模块化程度还是比较高的。这样的代码写得非常规范工整，值得好好地学习研究。

接下来的代码对模型的精度进行了验证：
``` python
acc1, acc5, loss = validate(config, data_loader_val, model)
logger.info(
    f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
)
max_accuracy = max(max_accuracy, acc1)
logger.info(f"Max accuracy: {max_accuracy:.2f}%")
```
`validate()`函数是这段代码最关键的函数。它的用法我暂且先不详细分析了。不过有一个新的用法需要注意：`@torch.no_grad()`语句的用法，参见[PyTorch官方torch.autograd.no_grad文档](https://pytorch.org/docs/1.7.1/autograd.html#torch.autograd.no_grad)。我们在空白脚本里测试一下下述代码：
``` python
import torch

x = torch.tensor([1.0], requires_grad=True)


def doubler(alpha):
    return alpha * 2


z = doubler(x)
print("x.requires_grad：", x.requires_grad)
print("z.requires_grad：", z.requires_grad)
```
结果为：
```
x.requires_grad： True
z.requires_grad： True
```
再在空白脚本中测试下面的代码：
``` python
import torch

x = torch.tensor([1.0], requires_grad=True)


@torch.no_grad()
def doubler(alpha):
    return alpha * 2


z = doubler(x)
print("x.requires_grad：", x.requires_grad)
print("z.requires_grad：", z.requires_grad)
```
结果为：
```
x.requires_grad： True
z.requires_grad： False
```
由此知，`@torch.no_grad()`这句话是为了使某个函数不计算梯度。如果一个函数前面加上了`@torch.no_grad()`这句话，则在这种模式下，每次计算的结果都是requires_grad=False，即使输入是requires_grad=True。`torch.no_grad`这个上下文管理器是线程本地的，它不会影响其他线程的计算。它也可以作为一个装饰器。此时，需要用括号将其实例化。详见[PyTorch官方torch.autograd.no_grad文档](https://pytorch.org/docs/1.7.1/autograd.html#torch.autograd.no_grad)。


最后的三行代码是打印出总的训练时间：
``` python
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
logger.info("Training time {}".format(total_time_str))
```

至此，Swin Transformer的训练脚本学习完毕。之后我会找时间学习一下Swin Transformer模型的设计。

（最后更新于2021.12.31）