import socket
import struct

SERVER_IP = "192.168.1.146"  # Unity 服务器的 IP 地址
SERVER_PORT = 5000  # Unity 服务端口

# 创建 TCP 客户端
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SERVER_IP, SERVER_PORT))
print(f"Connected to Unity server at {SERVER_IP}:{SERVER_PORT}.")

left = {}  # 用于存储左手数据
right = {}  # 用于存储右手数据 

# 确保接收完整的数据（包括数据包长度和实际数据）
def receive_data(socket, length):
    data = b""  # 创建一个空字节串
    while len(data) < length:
        packet = socket.recv(length - len(data))  # 接收剩余的字节
        if not packet:
            return None  # 没有数据时返回None，避免死循环
        data += packet
    return data

# 接收并处理数据的主函数
def process_data():
    while True:
        try:
            # 接受消息的长度(4个字节)，并确保接收到完整的长度字段
            length_data = b""
            while len(length_data) < 4:
                packet = client_socket.recv(4 - len(length_data))
                if not packet:
                    break
                length_data += packet

            # 使用 struct.unpack 获取消息长度  
            if len(length_data) < 4: 
                continue

            message_length = struct.unpack('< I', length_data)[0]
            print(f"Expected message length: {message_length}")

            # 根据接收到的长度接受实际数据
            data = receive_data(client_socket, message_length)

            if data:
                try:
                    # 将字节数据解码为字符串
                    data_str = data.decode("utf-8")
                    # print(f"Decoded data: {data_str}")

                    # 假设收到的数据格式为 handedness, joint, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w
                    parts = data_str.split(',')
                    if len(parts) != 9:
                        print("Error: Unexpected data format.")
                        continue  # 如果数据格式不对，跳过此次处理

                    handedness = parts[0].strip()  # 'Left' 或 'Right'
                    joint = parts[1].strip()  # 关节名称，如 'Wrist'
                    pos_x = float(parts[2].strip())
                    pos_y = float(parts[3].strip())
                    pos_z = float(parts[4].strip())
                    rot_x = float(parts[5].strip())
                    rot_y = float(parts[6].strip())
                    rot_z = float(parts[7].strip())
                    rot_w = float(parts[8].strip())

                    # 数据格式：['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']
                    joint_data = [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w]

                    # 根据 handedness 存储数据
                    if handedness == "Left":
                        left[joint] = joint_data
                    elif handedness == "Right":
                        right[joint] = joint_data

                    # 打印当前更新的字典
                    print(f"Left Hand: {left}")
                    print(f"Right Hand: {right}")

                except Exception as e:
                    print(f"Error processing data: {e}")
        except Exception as e:
            print(f"Error receiving data: {e}")

# 启动数据接收与处理
if __name__ == "__main__":
    process_data()
