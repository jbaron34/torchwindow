import torch
from torchwindow import Window

import time


if __name__ == "__main__":
    ones_v_half = torch.ones((300, 800, 1), dtype=torch.float32, device="cuda")
    zeros_v_half = torch.zeros((300, 800, 1), dtype=torch.float32, device="cuda")

    ones_h_half = torch.ones((600, 400, 1), dtype=torch.float32, device="cuda")
    zeros_h_half = torch.zeros((600, 400, 1), dtype=torch.float32, device="cuda")

    ones_whole = torch.ones((600, 800, 1), dtype=torch.float32, device="cuda")
    zeros_whole = torch.zeros((600, 800, 1), dtype=torch.float32, device="cuda")

    whole_v_half = torch.cat([ones_v_half, zeros_v_half], dim=0)
    whole_h_half = torch.cat([ones_h_half, zeros_h_half], dim=1)

    image = torch.cat([whole_v_half, zeros_whole, whole_h_half, ones_whole], dim=-1)

    window = Window(800, 600, "Example")
    window.draw(image)

    time.sleep(5)

    window.close()
