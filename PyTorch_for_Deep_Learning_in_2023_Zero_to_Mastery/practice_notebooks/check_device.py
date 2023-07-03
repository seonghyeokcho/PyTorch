import time

import torch
import platform


# mps 를 사용할 수 있는지 여부 확인하기
print(f"PyTorch version: {torch.__version__}") # 1.12.1 이상
print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}") # True 여야 합니다.
print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}") # True 여야 합니다.
print(platform.platform())

# cpu vs mph 속도 비교
def CPU_vs_GPU():
    # on CPU
    u = torch.rand(10000, 500)
    v = torch.rand(500, 10000)
    
    tic = time.time()
    torch.matmul(u, v)
    toc = time.time()
    print(f"{u.device} -> {toc - tic}")
    
    # on GPU
    u = torch.rand(10000, 500).to("mps")
    v = torch.rand(500, 10000).to("mps")
    
    tic = time.time()
    torch.matmul(u, v)
    toc = time.time()
    print(f"{u.device} -> {toc - tic}")


if __name__ == "__main__":
    CPU_vs_GPU()