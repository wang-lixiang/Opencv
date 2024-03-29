本项目要使用到face_recognition库，安装起来会比较麻烦：<br>
pip install CMake<br>
pip install dlib<br>
pip install face_recognition<br>
后面两个库的安装需要科学上网才行，读者也可以采用国内的镜像网站，笔者这里直接使用了科学上网<br>
这次的项目使用的是face_recognition库，它将自动帮助我们完成对人脸位置的定位以及
人脸的编码，我只需要对这一块进行调用即可。<br>
大致的思路是准备好已知的图片和label，然后视频检测展现出来。