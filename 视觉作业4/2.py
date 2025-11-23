import cv2
from ultralytics import YOLO

# --- 配置 ---
# 模型路径 (请修改为你实际的 best.pt 路径)
model_path = 'yolo11s.pt'
# 视频路径 (如果是摄像头，写 0)
video_path = 'video.mp4' 
# 要检测的目标类别名称 (根据你的 data.yaml)
target_class = 'water' 

def main():
    # 1. 加载模型
    print(f"正在加载模型: {model_path} ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"错误: 无法加载模型。请检查路径。\n{e}")
        return

    # 2. 打开视频源
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {video_path}")
        return

    print("开始检测... 按 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束。")
            break

        # 3. 使用 YOLO 进行预测
        # stream=True 可以减少内存占用
        results = model.predict(frame, conf=0.5, verbose=False)

        # 4. 处理每一帧的结果
        for result in results:
            # 获取检测到的框 (Boxes)
            boxes = result.boxes
            
            # 绘制结果到图像上 (YOLO 自带绘图功能)
            annotated_frame = result.plot()

            # 遍历每个框，查找矿泉水
            for box in boxes:
                # 获取类别 ID
                cls_id = int(box.cls[0])
                # 获取类别名称
                cls_name = model.names[cls_id]
                
                # 如果是目标类别 (water)
                if cls_name == target_class:
                    # 获取坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 计算中心点坐标
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 在终端打印
                    print(f"发现 {target_class}! 边界框: [{x1}, {y1}, {x2}, {y2}] | 中心点: ({center_x}, {center_y})")

        # 5. 显示图像窗口
        cv2.imshow('YOLO Detection', annotated_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()