# WSL2+GPU环境配置

## Win10预览版更新

想要在WSL2中配置GPU环境，第一步就是确定win10已开启预览版并更新至最新版本。具体而言：

- 在微软官网注册加入windows预览体验计划，[点此前往]([Windows Insider](https://insider.windows.com/en-us/getting-started#register))

- 在win10“设置”->“更新与安全”中找到“Windows预览体验计划”并开启，确认Microsoft账号后，选择“Dev渠道”

- 在win10“设置”->“更新与安全”中找到“Windows更新“，点击检查更新，安装预览版更新

- 使用快捷键win+r呼出运行，输入命令winver，查看当前内核版本，我个人的版本信息如下，当内核版本高于20150后可继续后续步骤。

  ![image-20210525192448298](D:\Projects\AI-LAB-Manual\img\image-20210525192448298.png)

**如果没有安装预览版更新并切换到Dev渠道，在后续安装完后会提示cannot load libcuda.so.1**

## WSL2环境配置

- 以管理员权限打开powershell，依次运行以下命令启用wsl服务与虚拟机功能：

```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

重启计算机，并执行后续步骤：

- 安装linux内核更新包：[点此下载](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)
- 将WSL2设置为默认版本：

```
wsl --set-default-version 2
```

- 安装Linux：

  - 在Microsoft Store中搜索对于的Linux发行版，如Ubuntu，选择合适的版本并安装
  - 启动安装的linux版本（第一次启动才相当于正式安装系统，会要求输入用户名并配置密码）

- 输入以下命令，确认启用WSL2

  ```
  wsl --list --verbose
  ```

**在文件管理器的地址栏中输入```\\wsl$``` 可快速前往linux安装路径，在wsl系统中使用```/mnt/盘符/```访问对应的磁盘**

参考文献：

[在 Windows 10 上安装 WSL | Microsoft Docs](https://docs.microsoft.com/zh-cn/windows/wsl/install-win10)

## GPU环境配置

0. 在WSL2安装好后，需要对linux系统初始化

   ```
   sudo apt-get update
   sudo apt-get install g++ gcc make
   ```

1. 然后为win10安装wsl的nvidia驱动，[点此下载](https://developer.nvidia.com/cuda/wsl)

2. 配置cuda和cudnn，配置方式和在普通ubuntu等linux系统的方式一致

   ubuntu中可使用apt-get安装cuda

   ```
   sudo apt-get install cuda
   ```

   然后在官网下载对于版本的cudnn，运行以下代码完成cudnn的安装(根据自身版本修改路径)

   ```
   tar xaf ~/Download/cudnn-10.0-linux-x64-v7.6.5.32.tgz
   sudo cp -Pv include/cudnn.h /usr/local/cuda-10.0/include/
   sudo cp -Pv lib64/* /usr/local/cuda-10.0/lib64/
   ```

3. 配置具体的GPU软件

**nvidia-smi命令在wsl中无法使用，仅可以在win10的终端中使用**

参考文献

[windows10 + wsl2,使用NVIDIA gpu_XXXXX的专栏-CSDN博客](https://blog.csdn.net/Tyronne/article/details/109319058)

[Enable NVIDIA CUDA in WSL 2 - Win32 apps | Microsoft Docs](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl)

