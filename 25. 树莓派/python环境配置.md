# Python环境配置

过程参考：https://blog.csdn.net/xuzhexing/article/details/99404943

## 1. 换中国源

查看树莓派版本：

```
lsb_release -c
```

根据版本选择中国源

换源方法：https://blog.csdn.net/la9998372/article/details/77886806

**Stretch版本：**

```sh
deb http://mirrors.aliyun.com/raspbian/raspbian/ stretch main contrib non-free rpi
```

**Buster版本：**

```sh
deb http://mirrors.aliyun.com/raspbian/raspbian/ buster main contrib non-free rpi
```

## 2.Vim 安装

```
sudo apt-get install -y vim
```

## 3. berryconda安装

https://github.com/jjhelmus/berryconda

[下载Berryconda3-2.0.0-Linux-armv7l.sh](http://xn--berryconda3-2-o40up927e.0.0-linux-armv7l.sh/)

```
chmod +x Berryconda3-2.0.0-Linux-armv7l.sh 
./Berryconda3-2.0.0-Linux-armv7l.sh
```

重启终端，python就更新到3.6版本了

 ## 4.安装相应的python包

可以使用pip或conda方式安装，但是安装速度很慢，需要耐心等待