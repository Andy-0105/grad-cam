import numpy as np
import cv2

DEBUG_COUNTER = 1

def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    # 進行奇異值分解
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        # activations - ndarray: (1280, 20, 20)
        # reshaped_activations - ndarray: (400, 1280)

        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()

        # # set minus value into zero
        # reshaped_activations = np.maximum(reshaped_activations, 0)

        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)

        # 對reshaped_activation進行奇異值分解 M=400, N=1280
        # U - 左奇異矩陣 ndarray: (400, 400)
        # S - 奇異值向量 ndarray: (400, )
        # VT - 右奇異矩陣 ndarray: (1280, 1280)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        # projection - ndarray: (400, )
        projection = reshaped_activations @ (VT[0, :])
        # projection - ndarray: (20, 20)
        projection = projection.reshape(activations.shape[1:])
        projection_img = projection * 255
        cv2.imwrite(f"debug/projection{DEBUG_COUNTER}.png", projection_img)
        cv2.imwrite(f"debug/resized_projection{DEBUG_COUNTER}.png", cv2.resize(projection_img * 255, (1280, 1280)))
        cv2.imwrite(f"debug/VT{DEBUG_COUNTER}.png", (VT - np.min(VT)) / np.max(VT) * 255)
        cv2.imwrite(f"debug/reshaped_activation{DEBUG_COUNTER}.png", (reshaped_activations - np.min(reshaped_activations)) / np.max(reshaped_activations) * 255)
        projections.append(projection)
    return np.float32(projections)
