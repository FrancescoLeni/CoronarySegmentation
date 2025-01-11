import numpy as np
import matplotlib.pyplot as plt
import cv2
import nrrd
import imageio

p_rec, _ = nrrd.read(r'C:\Users\franc\OneDrive\runs\reconstructed_preds_N20.nrrd')
x_rec, _ = nrrd.read(r'C:\Users\franc\OneDrive\runs\reconstructed_CT_N20.nrrd')
y_rec, _ = nrrd.read(r'C:\Users\franc\OneDrive\runs\reconstructed_masks_N20.nrrd')
dst = r'C:\Users\franc\OneDrive\runs'

p_rec = p_rec[15:146]
x_rec = x_rec[15:146]
y_rec = y_rec[15:146]
print(p_rec.shape, x_rec.shape, y_rec.shape)

# Video writer setup
frame_height, frame_width = 512, 512 # Dimensions of each slice
fps = 13  # Frames per second
# video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frames = []

# Create frames for each slice ( VS )
for now in range(x_rec.shape[0]):  # Iterate through all slices
    fig, ax = plt.subplots(1,1, figsize=(19.2, 10.8), dpi=100)

    # Overlay preparation
    overlay_pred = np.zeros((*p_rec[now].shape, 3))
    overlay_pred[:, :, 0] = p_rec[now]  # Red channel for the mask

    # overlay_gt = np.zeros((*y_rec[now].shape, 3))
    # overlay_gt[:, :, 0] = y_rec[now]  # Red channel for the mask

    # Plot the original slice with overlay
    ax.imshow(x_rec[now], cmap='gray')  # Grayscale background
    ax.imshow(overlay_pred, cmap='Reds', alpha=0.5)  # Red overlay for the mask
    ax.axis('off')  # Remove axes
    ax.set_title('Predicted mask', fontsize=20)

    # ax[1].imshow(x_rec[now], cmap='gray')  # Grayscale background
    # ax[1].imshow(overlay_gt, cmap='Reds', alpha=0.5)  # Red overlay for the mask
    # ax[1].axis('off')  # Remove axes
    # ax[1].set_title('Ground truth mask', fontsize=20)

    # Save the frame as an image
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Updated to use buffer_rgba
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Convert to proper shape (RGBA)
    frame = frame[:, :, :3]  # Drop the alpha channel to convert RGBA to RGB

    frames.append(frame)

    plt.close(fig)  # Close the figure to save memory

# Finalize and release the video writer
# video_writer.release()
imageio.mimwrite('pred_N20_10s.mp4', frames, fps=fps, codec='libx264')

print("Video saved as 'output_video.mp4'")
# frames = []
# # ONLY GT
# for now in range(x_rec.shape[0]):  # Iterate through all slices
#     fig, ax = plt.subplots(1,1, figsize=(19.2, 10.8), dpi=100)
#
#     overlay_gt = np.zeros((*y_rec[now].shape, 3))
#     overlay_gt[:, :, 0] = y_rec[now]  # Red channel for the mask
#
#     ax.imshow(x_rec[now], cmap='gray')  # Grayscale background
#     ax.imshow(overlay_gt, cmap='Reds', alpha=0.5)  # Red overlay for the mask
#     ax.axis('off')  # Remove axes
#     ax.set_title('Ground truth mask', fontsize=20)
#
#     # Save the frame as an image
#     fig.canvas.draw()
#     frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Updated to use buffer_rgba
#     frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Convert to proper shape (RGBA)
#     frame = frame[:, :, :3]  # Drop the alpha channel to convert RGBA to RGB
#
#     frames.append(frame)
#
#     plt.close(fig)  # Close the figure to save memory
#
# # Finalize and release the video writer
# # video_writer.release()
# imageio.mimwrite(dst+'\gt_N20_10s.mp4', frames, fps=fps, codec='libx264')
#
# frames = []
# # ONLY Pred
# for now in range(x_rec.shape[0]):  # Iterate through all slices
#     fig, ax = plt.subplots(1,1, figsize=(19.2, 10.8), dpi=100)
#
#     overlay_pred = np.zeros((*p_rec[now].shape, 3))
#     overlay_pred[:, :, 0] = p_rec[now]  # Red channel for the mask
#
#     ax.imshow(x_rec[now], cmap='gray')  # Grayscale background
#     ax.imshow(overlay_pred, cmap='Reds', alpha=0.5)  # Red overlay for the mask
#     ax.axis('off')  # Remove axes
#     ax.set_title('Predicted mask', fontsize=20)
#
#     # Save the frame as an image
#     fig.canvas.draw()
#     frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # Updated to use buffer_rgba
#     frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # Convert to proper shape (RGBA)
#     frame = frame[:, :, :3]  # Drop the alpha channel to convert RGBA to RGB
#
#     frames.append(frame)
#
#     plt.close(fig)  # Close the figure to save memory
#
# # Finalize and release the video writer
# # video_writer.release()
# imageio.mimwrite(dst+'\pred_N20_10s.mp4', frames, fps=fps, codec='libx264')