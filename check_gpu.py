import torch
# Câu lệnh then chốt để kiểm tra
if torch.cuda.is_available():
    print("✅ Chúc mừng! PyTorch đang sử dụng GPU.")
    print("--- Chi tiết GPU ---")
    
    # Lấy số lượng GPU
    print(f"Số lượng GPU: {torch.cuda.device_count()}")
    
    # Lấy tên của GPU (thiết bị 0)
    print(f"Tên GPU (Thiết bị 0): {torch.cuda.get_device_name(0)}")
    
else:
    print("⚠️ Cảnh báo! PyTorch đang sử dụng CPU.")