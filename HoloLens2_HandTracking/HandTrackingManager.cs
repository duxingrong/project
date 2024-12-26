using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.Utilities;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;  // 添加异步支持
using System;

public class HandTrackingManager : MonoBehaviour
{
    // TCP 通信
    private TcpListener tcpListener;
    private TcpClient tcpClient;
    private NetworkStream networkStream;
    public int serverPort = 5000;

    // 变量
    public GameObject cubePrefab; // 黑色小方块的预制体
    private Dictionary<TrackedHandJoint, GameObject> leftHandJoints = new Dictionary<TrackedHandJoint, GameObject>();
    private Dictionary<TrackedHandJoint, GameObject> rightHandJoints = new Dictionary<TrackedHandJoint, GameObject>();

    // 网络连接状态
    private bool isConnected = false;
    private Queue<string> sendQueue = new Queue<string>();  // 用于存放待发送的数据
    private Thread sendThread;  // 数据发送线程
    private CancellationTokenSource cancellationTokenSource;  // 用于控制线程的取消

    void Start()
    {
        // 初始化关节的 Cube
        foreach (TrackedHandJoint joint in Enum.GetValues(typeof(TrackedHandJoint)))
        {
            // 左手
            GameObject leftCube = Instantiate(cubePrefab);
            leftCube.transform.localScale = Vector3.one * 0.01f;
            leftCube.GetComponent<Renderer>().material.color = Color.red;
            leftHandJoints[joint] = leftCube;

            // 右手
            GameObject rightCube = Instantiate(cubePrefab);
            rightCube.transform.localScale = Vector3.one * 0.01f;
            rightCube.GetComponent<Renderer>().material.color = Color.red;
            rightHandJoints[joint] = rightCube;
        }

        // 初始化 TCP 服务器
        try
        {
            tcpListener = new TcpListener(IPAddress.Any, serverPort);
            tcpListener.Start();
            Debug.Log($"Server started on port {serverPort}. Waiting for a connection...");

            tcpListener.BeginAcceptTcpClient(OnClientConnected, null);
        }
        catch (Exception ex)
        {
            Debug.LogError($"Failed to start server: {ex.Message}");
        }

        // 启动数据发送线程
        cancellationTokenSource = new CancellationTokenSource();
        sendThread = new Thread(() => DataSendThread(cancellationTokenSource.Token));
        sendThread.Start();
    }

    void Update()
    {
        // 更新左手关节位置
        UpdateHandJoints(Handedness.Left, leftHandJoints);

        // 更新右手关节位置
        UpdateHandJoints(Handedness.Right, rightHandJoints);
    }

    private void UpdateHandJoints(Handedness handedness, Dictionary<TrackedHandJoint, GameObject> handJoints)
    {
        foreach (TrackedHandJoint joint in Enum.GetValues(typeof(TrackedHandJoint)))
        {
            MixedRealityPose pose;
            bool isTracked = HandJointUtils.TryGetJointPose(joint, handedness, out pose);

            if (isTracked)
            {
                // 更新方块位置
                handJoints[joint].transform.position = pose.Position;
                handJoints[joint].transform.rotation = pose.Rotation;

                // 如果网络连接可用，发送数据
                if (isConnected)
                {
                    string jointData = $"{handedness},{joint},{pose.Position.x},{pose.Position.y},{pose.Position.z}," +
                        $"{pose.Rotation.x},{pose.Rotation.y},{pose.Rotation.z},{pose.Rotation.w}";

                    // 将数据添加到发送队列
                    lock (sendQueue)
                    {
                        sendQueue.Enqueue(jointData);
                    }
                }
            }
        }
    }

    private void DataSendThread(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            // 检查是否有待发送的数据
            string jointData = null;
            lock (sendQueue)
            {
                if (sendQueue.Count > 0)
                {
                    jointData = sendQueue.Dequeue();
                }
            }

            if (!string.IsNullOrEmpty(jointData) && networkStream != null && networkStream.CanWrite)
            {
                try
                {
                    // 将数据转换为字节数组
                    byte[] data = Encoding.UTF8.GetBytes(jointData);
                    byte[] lengthData = BitConverter.GetBytes(data.Length);

                    // 发送数据
                    networkStream.Write(lengthData, 0, lengthData.Length);
                    networkStream.Write(data, 0, data.Length);
                    Debug.Log($"Sent: {jointData}");
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error sending data: {ex.Message}");
                }
            }

            // 休眠一小段时间，避免过度占用 CPU
            Thread.Sleep(10);
        }
    }

    private void OnClientConnected(IAsyncResult result)
    {
        try
        {
            tcpClient = tcpListener.EndAcceptTcpClient(result);
            networkStream = tcpClient.GetStream();
            isConnected = true;
            Debug.Log("Client connected.");

            // 确保重新连接时能够正确接收数据
            if (networkStream != null && networkStream.CanWrite)
            {
                Debug.Log("Network stream is ready.");
            }

            // 继续监听其他客户端连接
            tcpListener.BeginAcceptTcpClient(OnClientConnected, null);
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error in client connection: {ex.Message}");
        }
    }

    private void OnApplicationQuit()
    {
        // 停止数据发送线程
        cancellationTokenSource.Cancel();
        sendThread.Join();

        // 清理资源
        if (networkStream != null)
        {
            networkStream.Close();
            Debug.Log("Network stream closed.");
        }

        if (tcpClient != null)
        {
            tcpClient.Close();
            Debug.Log("TCP client closed.");
        }

        if (tcpListener != null)
        {
            tcpListener.Stop();
            Debug.Log("TCP listener stopped.");
        }
    }
}
