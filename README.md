# st-convert

把 RWKV 的 `.pth`（torch zip checkpoint）转成 Ai00 用的 `.safetensors`。

输出与 Python 参考实现（`safetensors.torch.save_file` 布局）**逐字节一致**：
header JSON 以空格补齐到 8 字节、张量按 key 排序、数据统一转成 fp16。

## 构建

```powershell
cargo build --release
```

无第三方依赖（zip 中央目录、torch pickle、safetensors 头都是手写的），
所以不需要联网、也不会被依赖版本漂移影响。

## 用法

```powershell
st-convert.exe --input "E:\Aria2Download\rwkv7-g1l-13.3b-20261031-ctx34304.pth" `
               --output "D:\LLM\Ai00\Models\rwkv7-g1l-13b3.st"
```

| 参数 | 说明 |
|---|---|
| `--input` | 输入的 `.pth` |
| `--output` | 输出的 `.st`（同目录会自动创建） |
| `--batch-mb` | 每次顺序读取的块大小；`0`（默认）按空闲内存自动算 |
| `--threads` | 转换/写入线程数；`0`（默认）`min(8, 核心数)` |

默认批量大小 = `clamp(空闲内存 / 8 / (线程数 + 2), 32 MiB, 256 MiB)`，
并保证不小于最大的张量。

## 结构

```
src/main.rs    参数、plan、header、流水线
src/zip.rs     zip 目录 + zip64 + stored 条目读取
src/pickle.rs  torch state_dict 的最小 pickle 虚拟机
src/half.rs    bf16/f32 -> fp16（RNE，和 torch.half() 一致）
src/sysio.rs   Windows 定位读写（seek_read/seek_write）+ 空闲内存查询
```

流水线：1 个 reader 线程顺序大块读取，`--threads` 个 worker 线程
`frombuffer -> fp16 -> 转置 -> seek_write` 到最终偏移；缓冲区通过
空闲池复用，最多 `线程数 + 2` 块同时存在。

torch 把所有权重塞进少数几个大 storage（每个对应 zip 里的一个 `data/N`
条目），每个张量都是它的 view，所以源偏移 = `storage 载荷偏移 +
tensor.storage_offset * 元素大小`，不需要把整个模型读进内存。

## 校验

在四个 RWKV v7 (g1l) checkpoint 上与 Python 版本输出做了 md5 对比，全部一致：

| 模型 | 输出 | md5 |
|---|---|---|
| 1.5b | rwkv7-g1l-1b5.st | `3f41dee2…c35c4a22` |
| 2.9b | rwkv7-g1l-2b9.st | `f3bf0a87…b55bf2611` |
| 7.2b | rwkv7-g1l-7b2.st | `e9dd8fbb…cbcfd48ce` |
| 13.3b | rwkv7-g1l-13b3.st | `cc8fce16…d18e8d802` |

`cargo test` 里还有 fp16 转换的边界用例（inf/nan、溢出到 inf、最小正常数、
最小次正规数、半路值 ties-to-even）。

## 实测

同一时刻、同一块盘上的对比（磁盘写速会随连续写入量掉速，所以只放同批次数据）：

| 模型 | Rust | Python | Rust 峰值内存 |
|---|---|---|---|
| 1.5b | **18.4 s** | 30.5 s | 3.3 GiB |
| 2.9b | **36–42 s** | 49.6 s | 4.8 GiB |
| 7.2b | **48.7 s** | 53.1 s | ~5 GiB |
| 13.3b | 108–206 s | — | 5.6 GiB |

13.3b 的波动来自 SSD：连续写了几百 GB 测试文件后，写入速度从 ~2.3 GB/s
掉到 ~150 MB/s，同一台机器上的 Python 版也会同样变慢。即使在最慢的
206 s 那一次，md5 依然和 Python 输出完全一致。

## 限制

- 只支持 STORED（未压缩）zip 条目 —— torch 默认就是 STORED。
- 只支持连续（row-major）张量，遇到带 stride 的 view 会直接报错。
- 老 v5.1 checkpoint（带 `gate.weight`、需要把 `time_decay/time_faaaa`
  广播展开）未实现，会明确报错而不是输出错误文件。
- 需要 64 位且有足够地址空间；单文件 >4 GiB 走 zip64。

