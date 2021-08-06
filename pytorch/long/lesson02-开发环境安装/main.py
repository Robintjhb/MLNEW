import  torch

print(torch.__version__) #torch 版本
print('gpu:', torch.cuda.is_available()) #查看是否支持gpu