# 基于LAMA的图像修复界面
## 界面效果展示😊
### Demo支持功能：图像加载；图像修复；掩码笔触尺寸修改；自定义掩码；保存图像。  
![1.png](https://github.com/1024803482/Inpainting-Demo-Lama/blob/master/1.png)  
![2.png](https://github.com/1024803482/Inpainting-Demo-Lama/blob/master/2.png)
## 基本环境搭建
基本环境搭建可参考[LAMA](https://github.com/saic-mdal/lama)算法，这是一个优秀的基于深度学习的图像修复算法。我们使用big-lama作为基本模型，下载及描述信息可以在[LAMA](https://github.com/saic-mdal/lama)中进行查阅。除此之外，界面开发需要安装PyQt5, imageio库:  
  ```
  # Download libs
  pip install PyQt5
  pip install imageio
  ```
## 需要修改的部分
相较于LAMA，我们做了一些修改以适应界面搭建，相关修改在项目中：  
### 新建in_dir(./temp),和out_dir(./lama)两个文件夹  
### 修改配置文件  
"config/prediction/default.yaml"中in_dir, out_dir和path改成in_dir,out_dir和checkpoint文件
### 对GUI内的相关路径进行修正
### 运行GUI，完成Demo展示
