import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 假设我们有一个简单的模型和优化器
model = torch.nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 配置学习率调度器
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

# 设置预热学习率
initial_lr = optimizer.param_groups[0]['lr']
warmup_lr_schedule = torch.optim.lr_scheduler.LambdaLR(
    optimizer, 
    lr_lambda=lambda step: (step / lr_config['warmup_iters']) * (1 - lr_config['warmup_ratio']) + lr_config['warmup_ratio']
)

# 设置余弦退火学习率调度器
cosine_annealing_schedule = CosineAnnealingLR(
    optimizer, 
    T_max=1000,  # 通常是总的训练迭代次数
    eta_min=initial_lr * lr_config['min_lr_ratio']
)

# 综合调度器，可以在预热结束后切换到余弦退火调度器
class WarmupCosineAnnealingScheduler:
    def __init__(self, optimizer, warmup_scheduler, cosine_scheduler, warmup_iters):
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.cosine_scheduler = cosine_scheduler
        self.warmup_iters = warmup_iters
        self.step_num = 0
    
    def step(self):
        if self.step_num < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
        self.step_num += 1

# 创建综合调度器
scheduler = WarmupCosineAnnealingScheduler(optimizer, warmup_lr_schedule, cosine_annealing_schedule, lr_config['warmup_iters'])

# 模拟训练过程
num_epochs = 10
for epoch in range(num_epochs):
    for batch in range(100):
        # 假设我们有训练代码
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        
        # 更新学习率
        scheduler.step()
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, Batch {batch}, LR: {current_lr}")
