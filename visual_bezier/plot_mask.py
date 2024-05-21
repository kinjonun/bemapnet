import numpy as np
import matplotlib.pyplot as plt

# 创建一个示例的灰度图像（二维数组），数值范围从0到1
image = np.random.rand(10, 10)

# 显示灰度图像
plt.imshow(image, cmap='gray')
plt.title('Gray Scale Image')
plt.colorbar()  # 显示颜色条
plt.axis('off')  # 隐藏坐标轴
plt.show()
