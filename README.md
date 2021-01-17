# minifly upper computer
 接收minifly回传图像、可进行飞行控制的上位机
 
## 串口通信
在编写串口通信工具中，需要实现一个函数，自动找到对应com 口，并且连接该com口，保证后续通信正常作为初始化过程的一部分。

思路：
在win 设备管理器中，经常会出现多个com 口，但并不是每个com 口都是目标设备所链接的。
尝试打开每个com 口，输入enter 按键， 正确的com 口，会有ack log 返回，表明通信正常否则，没有任何log 返回，则判断为非目标设备所连接的com 口。

## 读取图像
我们可以发送HTTP请求给指定的URL（统一资源定位符），这个URL就是所谓的网络API，如果请求成功，它会返回HTTP响应，而HTTP响应的消息体中就有我们需要的Jpg格式的数据。

http://192.168.1.1:80/snapshot.cgi?resolution=11&user=admin&pwd=

### WiFi连接
在WINDOWS的WIFI列表中手动切换WIFI时，系统会先断开当前WIFI连接，再连接等待连接的WIFI。在程序中搜索到minifly的WIFI后，不断开当前WIFI连接，直接与minifly建立连接，无需等待，可瞬间完成连接，节约了大约八秒的等待时间。
