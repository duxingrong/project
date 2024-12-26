"""
主线程: 负责将数据放入队列中
子线程: 负责从队列中取数据并通过TCP socket 发送出去
线程间的同步: 使用queue.Queue 来作为线程安全的队列
"""

import socket 
import threading
import queue #提供了内置的锁,确保队列的操作是线程安全的
import time 
import struct 

class SocketServer():
    def __init__(self,host="127.0.0.1",port=5000):
        self.host = host 
        self.port = port
        self.send_queue = queue.Queue() #创建一个线程安全的队列
        self.server_socket = None
        self.client_socket = None 
        self.if_running = False 

    
    def start_server(self):
        """启动服务器"""
        self.server_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.server_socket.bind((self.host,self.port)) #元组
        self.server_socket.listen(1) #只允许一个客户端连接
        print(f"Server Start at {self.host}:{self.port}")

        #等待客户端连接
        self.client_socket,addr = self.server_socket.accept() #返回新的socket对象和对象的地址
        print(f"Client connected from {addr}")

        self.is_running = True 

        #创建并启动发送数据的线程
        send_thread = threading.Thread(target = self.send_data_thread)
        send_thread.daemon = True #设置为守护线程,主线程退出时子线程也会退出
        send_thread.start() 

    
    def send_data_thread(self):
        """从队列中取出元素发送出去"""
        while self.is_running:
            try:
                #从队列中取数据(如果队列为空,阻塞等待)
                data = self.send_queue.get(block=True,timeout=1)
                
                if data :
                    #发送数据
                    self.send_data(data)
                    print(f"Sent:{data}")

            except queue.Empty:
                continue #队列为空,继续等待

    def send_data(self,data):
        """发送数据到客户端"""
        if self.client_socket:
            try:


                #将数据转换成字节并发送
                byte_data = data.encode("utf-8")
                len_byte_data = struct.pack("<I",len(byte_data)) #使用struct打包成4字节
                self.client_socket.sendall(len_byte_data)
                self.client_socket.sendall(byte_data)

            except Exception as e:
                print(f"Error sending data :{e}")
                self.stop_server()

    def put_data_in_queue(self,data):
        """将数据放入队列,主线程调用"""
        self.send_queue.put(data)

    def stop_server(self):
        """停止服务器"""
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        self.is_running= False 
        print("Server stoped.")



if __name__=="__main__":
    server = SocketServer()
    server.start_server()

    #模拟向队列中放数据
    for i in range(5):
        server.put_data_in_queue(f"Message {i}")
        time.sleep(1)
        
    server.stop_server()
