#!/home/waleed/anaconda3/envs/tf/bin/python

import os
import cv2
import pickle


def main():

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data",)
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.join(DATA_DIR, f"training_data.pkl")

    print(f'Loading dataset: {filename}')

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    images = data['images']
    labels = data['labels']

    print('======================================')
    print(' Training Data Viewer')
    print('======================================')
    print(f'Number of images : {len(images)}')
    print(f'Number of labels : {len(labels)}')
    print('======================================')

    if len(images) != len(labels):
        print('WARNING: Image/label count mismatch!')


    for i, image in enumerate(images):

        # Get corresponding velocity
        linear_x = labels[i][0]
        angular_z = labels[i][1]

        # Print values
        print(f'Sample {i:05d} | ' f'linear.x = {linear_x:.4f} | ' f'angular.z = {angular_z:.4f}')

        # Add velocity information to image
        display_image = image.copy()

        cv2.putText(display_image, f'Sample: {i}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_image, f'linear.x: {linear_x:.3f}', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_image, f'angular.z: {angular_z:.3f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show image
        cv2.imshow('Training Data', display_image)

        # -----------------------------------------
        # Keyboard controls
        #
        # q / ESC = quit
        # any other key = next image
        # -----------------------------------------

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:
            print('Viewer stopped.')
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()