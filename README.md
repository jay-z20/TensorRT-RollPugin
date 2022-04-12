

## swin-transformer roll插件实现

C++ Tensorrt 实现 `pytroch` 中  `torch.roll()` 


### 参数


> // 保存 engine 
> 
> roll.exe -s -shift 1 1 -dims 1 2
> 
> // 推理
> 
> roll.exe -shift 1 1 -dims 1 2

### 测试

输入矩阵 3x4x5


