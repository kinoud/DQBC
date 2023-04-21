import torch
import torch.nn as nn
import torch.nn.functional as F
from .IFNet import IFBlock
from ..warplayer import warp
from validate.bucket import Bucket

class FlowTeacher(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.block_tea = IFBlock(17+4, c=cfg.c, mask=cfg.recalc_mask)
        self.recalc_mask = cfg.recalc_mask
        self.margin = cfg.margin
        self.w_tea = cfg.w_tea
    
    def forward(self, im0,im1,flow,mask,gt, student_flow_list,bkt:Bucket=None):
        wp0 = warp(im0, flow[:,0:2])
        wp1 = warp(im1, flow[:,2:4])

        inp = torch.cat((im0, im1, wp0, wp1, mask, gt), 1)
        
        output_tea = self.block_tea(inp.detach(), flow.detach(), scale=1)

        if self.recalc_mask:
            flow_d, mask_d = output_tea
        else:
            flow_d = output_tea
            mask_d = 0

        flow_teacher = flow.detach() + flow_d
        mask_teacher = mask.detach() + mask_d
        warped_img0_teacher = warp(im0, flow_teacher[:, :2])
        warped_img1_teacher = warp(im1, flow_teacher[:, 2:4])

        mask_teacher = torch.softmax(mask_teacher,1)
        merged_teacher = warped_img0_teacher * mask_teacher[:,[0]] + warped_img1_teacher * mask_teacher[:,[1]]
        
        loss_distill = 0

        if bkt:
            bkt.new_bi_flow(flow_teacher,'flow')
            bkt.new_gray(mask_teacher[:,[0]],'mask')
            bkt.new_rgb(merged_teacher,'merged')

        mask_stu = torch.softmax(mask,1)
        assert len(student_flow_list)==len(self.w_tea)
        for i, ((flow, scale),w) in enumerate(zip(student_flow_list,self.w_tea)):
            if w==0:
                continue
            
            flow = F.interpolate(flow, scale_factor = scale, mode="bilinear", align_corners=False) * scale
            wp0 = warp(im0, flow[:,0:2])
            wp1 = warp(im1, flow[:,2:4])

            merged = wp0*mask_stu[:,[0]]+wp1*mask_stu[:,[1]]

            loss_mask = ((merged - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + self.margin).float().detach()
            loss_distill += w*(((flow_teacher.detach() - flow) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()

            if bkt:
                bkt.new_rgb(merged, f'merged_stu{i}')
                bkt.new_gray(loss_mask, f'loss_mask{i}')
        
        outputs = {
            'loss_distill': loss_distill,
            'merged_tea': merged_teacher
        }

        return outputs
