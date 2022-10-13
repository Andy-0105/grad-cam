import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM, EigenGradCAM, GradCAM, GradCAMPlusPlus, GradCAMElementWise, XGradCAM, ScoreCAM, AblationCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import os
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.experimental import attempt_load
from utils.general import non_max_suppression

COLORS = [(0, 0, 255)]

device = torch.device("cuda:1")

ADD_PADDING = True
padding_size = 25

def parse_detections(results):
    class_names = ["fracture"]
    boxes, colors, names = [], [], []

    for i, detection in enumerate(results):
        confidence = detection[4]
        if confidence < 0.2:
            continue
        xmin = int(detection[0])
        ymin = int(detection[1])
        xmax = int(detection[2])
        ymax = int(detection[3])
        name = class_names[int(detection[5])]
        category = int(detection[5])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img

def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=False)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes

def zero_out_boxes(boxes, grayscale_cam):
    zero_out_cam = np.ones(grayscale_cam.shape, dtype=np.float32)
    H, W = grayscale_cam.shape
    for x1, y1, x2, y2 in boxes:
        if ADD_PADDING:
            padding_x1 = max(x1 - padding_size, 0)
            padding_x2 = min(x2 + padding_size, W - 1)
            padding_y1 = max(y1 - padding_size, 0)
            padding_y2 = min(y2 + padding_size, H - 1)
            mean = grayscale_cam[y1:y2, x1:x2].mean()
            # for i in range(padding_size):
            #     zero_out_cam[padding_y1+i:padding_y2-i, padding_x1+i:padding_x2-i] = 1 - (mean / 4) * ((i + 1) / padding_size)

            for i in range(padding_size):

                zero_out_cam[padding_y1 + i:padding_y2 - i, padding_x1 + i:padding_x2 - i] = 1 - ( i / padding_size ) * (1 - grayscale_cam[padding_y1 + i:padding_y2 - i, padding_x1 + i:padding_x2 - i])

        zero_out_cam[y1:y2, x1:x2] = grayscale_cam[y1:y2, x1:x2].copy()
    return zero_out_cam

model = attempt_load("w6_48times_K_Fold_0.pt", map_location=device)
model.eval()
model.to(device=device)
# target_layers = [*(model.model._modules['122']._modules['m2']._modules[f'{i}'] for i in range(4))]
target_layers = [*(model.model[i] for i in (-4, -3))]
# targets = [ClassifierOutputTarget(0)]

cam = EigenCAM(model, target_layers)

image_dir = "images"

for image_url in os.listdir(image_dir):
    image_url = os.path.join(image_dir, image_url)
    image_filename, image_ext = os.path.splitext(os.path.basename(image_url))
    img = cv2.imread(image_url)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1280, 1280))
    rgb_img = img.copy()
    img = np.float32(img) / 255
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0).to(device=device)

    preds = model(tensor)[0]
    results = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.3)[0]
    boxes, colors, names = parse_detections(results)
    detections = draw_detections(boxes, colors, names, rgb_img.copy())

    grayscale_cam = cam(tensor)[0, :, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=False, image_weight=.5)
    cam_image = draw_detections(boxes, colors, names, cam_image.copy())

    zero_out_grayscale_cam = zero_out_boxes(boxes, grayscale_cam)
    zero_out_cam_image = show_cam_on_image(img, zero_out_grayscale_cam, use_rgb=False, image_weight=.5)
    zero_out_cam_image = draw_detections(boxes, colors, names, zero_out_cam_image.copy())
    Image.fromarray(zero_out_cam_image).save(f"outputs/{image_filename}_cam{image_ext}")

    Image.fromarray(np.hstack((rgb_img, cam_image, zero_out_cam_image))).save(f"outputs/stacks/{image_filename}_stack{image_ext}")
    # renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)
    # Image.fromarray(renormalized_cam_image).save(f"outputs/{image_filename}_renormalized{image_ext}")
    #
    # Image.fromarray(np.hstack((rgb_img, detections, cam_image, renormalized_cam_image))).save(f"outputs/stacks/{image_filename}_stack{image_ext}")