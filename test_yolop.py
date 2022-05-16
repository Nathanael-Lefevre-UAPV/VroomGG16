import torch

if __name__ == "__main__":
    # load model
    model = torch.hub.load('hustvl/yolop', 'yolop', trust_repo=True, pretrained=True)

    #inference
    img = torch.randn(1,3,640,640)
    det_out, da_seg_out,ll_seg_out = model(img)