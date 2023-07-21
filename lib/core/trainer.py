import logging
import time
import torch

class Trainer(object):
    def __init__(self, cfg, model, rank, output_dir, writer_dict, awl):
        self.model = model
        self.output_dir = output_dir
        self.rank = rank
        self.print_freq = cfg.PRINT_FREQ
        self.is_awl = cfg.AWL
        self.awl = awl

    def train(self, epoch, data_loader, optimizer):
        logger = logging.getLogger("Training")

        batch_time = AverageMeter()
        data_time = AverageMeter()
        multi_heatmap_loss_meter = AverageMeter()
        single_heatmap_loss_meter = AverageMeter()
        contrastive_loss_meter = AverageMeter()
        inst_loss_meter = AverageMeter()
        semantic_loss_meter = AverageMeter()
        pixel_loss_meter = AverageMeter()

        self.model.train()

        end = time.time()
        for i, batched_inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            loss_dict = self.model(batched_inputs)

            loss = 0
            num_images = len(batched_inputs)
            if 'multi_heatmap_loss' in loss_dict:
                multi_heatmap_loss = loss_dict['multi_heatmap_loss']
                multi_heatmap_loss_meter.update(multi_heatmap_loss.item(), num_images)
                loss += multi_heatmap_loss

            if 'single_heatmap_loss' in loss_dict:
                single_heatmap_loss = loss_dict['single_heatmap_loss']
                single_heatmap_loss_meter.update(single_heatmap_loss.item(), num_images)
                loss += single_heatmap_loss
            
            if 'contrastive_loss' in loss_dict:
                contrastive_loss = loss_dict['contrastive_loss']
                contrastive_loss_meter.update(contrastive_loss.item(), num_images)
                loss += contrastive_loss
            
            if 'inst_loss' in loss_dict:
                inst_loss = loss_dict['inst_loss']
                inst_loss_meter.update(inst_loss.item(), num_images)
                loss += inst_loss
                
            if 'semantic_loss' in loss_dict:
                semantic_loss = loss_dict['semantic_loss']
                semantic_loss_meter.update(semantic_loss.item(), num_images)
                loss += semantic_loss
            
            if 'pixel_loss' in loss_dict:
                pixel_loss = loss_dict['pixel_loss']
                pixel_loss_meter.update(pixel_loss.item(), num_images)
                loss += pixel_loss
                
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 and self.rank == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      '{multiple}{single}{contrast}{inst}{semantic}{pixel}'.format(
                        epoch, i, len(data_loader),
                        batch_time=batch_time,
                        speed=num_images / batch_time.val,
                        data_time=data_time,
                        multiple=_get_loss_info(multi_heatmap_loss_meter, 'multiple'),
                        single=_get_loss_info(single_heatmap_loss_meter, 'single'),
                        contrast=_get_loss_info(contrastive_loss_meter, 'contrast'),
                        inst=_get_loss_info(inst_loss_meter, 'inst'),
                        semantic=_get_loss_info(semantic_loss_meter, 'semantic'),
                        pixel=_get_loss_info(pixel_loss_meter, 'pixel')
                    )
                logger.info(msg)

def _get_loss_info(meter, loss_name):
    msg = '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(name=loss_name, meter=meter)
    return msg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count if self.count != 0 else 0