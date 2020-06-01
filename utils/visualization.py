import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from matplotlib import patches
from torchvision import transforms

from utils.bbox import yolo_to_coco
from utils.data_loading import INV_NORMALIZE

TO_PIL = transforms.ToPILImage()
TO_TENSOR = transforms.ToTensor()


def image_bbox_vis(image, preds):
    bbox_pred = preds[:4]
    objectness_pred = preds[4]

    coco_bbox_pred = yolo_to_coco(bbox_pred)
    objectness_pred_list = objectness_pred.view(-1)

    ad = AnnotationDrawer(image)
    ad.draw_bboxes(coco_bbox_pred, objectness=objectness_pred_list)
    vis_image = ad.get_tensor()

    return vis_image


def batch_bbox_vis(images, preds):
    return torch.stack([image_bbox_vis(images[i], preds[i]) for i in range(preds.shape[0])])


class AnnotationDrawer:
    def __init__(self, tensor):
        self.tensor = INV_NORMALIZE(tensor)
        self.image = TO_PIL(self.tensor)
        self.draw = ImageDraw.Draw(self.image, 'RGBA')

    def show_image(self):
        plt.imshow(self.image)
        plt.show()

    def draw_bboxes(self, bboxes, objectness=None, color=(255, 0, 0)):
        for i, bbox in enumerate(bboxes):
            # TODO: replace with helper function
            x0, y0, w, h = bbox
            x1, y1 = x0 + w, y0 + h

            alpha = 100
            if objectness is not None:
                alpha = 255 * objectness[i]

            self.draw.rectangle([x0, y0, x1, y1], outline=(*color, alpha), width=1)

    def get_tensor(self):
        return TO_TENSOR(self.image)

    def draw_objectness(self, objectness):
        original_size = self.tensor.shape[1:]
        objectness_image = TO_PIL(objectness)
        objectness_image = objectness_image.resize(original_size, resample=Image.BOX)
        objectness_image = objectness_image.convert("RGB")

        self.image = Image.blend(self.image, objectness_image, 0.2)
