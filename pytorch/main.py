import torch


def print_hi(name):
    print(f'Hi, {name}')
    print(torch.__version__) #torch 版本
    print(torch.cuda.is_available()) #是否支持gpu


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
