"""
用于测试SocketServer()类
"""
import socket 
import struct 

if __name__=="__main__":
    host = '127.0.0.1'
    port = 5000
    client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    
    try:
        #连接到服务器
        client_socket.connect((host,port))
        print(f"Connected to Unity server at {host}:{port}.")

        while True:
            #接受数据长度(4个字节)
            length_byte_data = b""
            while len(length_byte_data)<4:
                packet  = client_socket.recv(4-len(length_byte_data))
                if not packet :
                    break #如果没有数据，表示连接关闭
                length_byte_data +=packet
            
            #解析数据长度
            message_length = struct.unpack('<I',length_byte_data)[0] #使用struct.unpack获取长度
            print(f"Expecting message length:{message_length}")

            #接受真实数据
            data = b""
            while len(data)<message_length:
                packet = client_socket.recv(message_length-len(data))
                if not packet:
                    break 
                data+=packet
            
            if data:
                #在这里增加对data数据的处理
                message = data.decode('utf-8')
                print(f"Received data :{message}")

    except Exception as e:
        print(f"Error:{e}")

    finally:
        client_socket.close()
        print("Connection closed.")
            
