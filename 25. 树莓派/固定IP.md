# 固定IP

```
sudo nano /etc/dhcpcd.conf
```

定位到以下位置：

![在这里插入图片描述](../img/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjEwODQ4NA==,size_16,color_FFFFFF,t_70)

将该部分取消注释

## 1.固定无线

其中：wlan0代表无线，也就是指定接口
**ip_address**代表设置的静态ip地址
routers代表路由器/网关IP地址   需要与ifconfig中broadcast对应，否则无法连接外网

 ## 2.固定有限

