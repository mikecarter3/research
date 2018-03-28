# USAGE
# python download_images.py --urls urls.txt --output images/santa
# if --qa is specified, the images will be proposed one by one and you
# will be prompted to keep (by pressing 'k' key) or else the image will
# be deleted.

# Credits to https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
# for the starter code on this utility.

from imutils import paths
import argparse
import requests
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
    help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True,
    help="path to output directory of images")
ap.add_argument('-q', "--qa", required=False, action='store_true', default=False,
    help='set this flag if you want to manually QA images')
ap.add_argument('-n', "--only_qa", required=False, action='store_true', default=False,
    help='set this flag if you only want to do qa, not redownload all pictures.')
args = vars(ap.parse_args())

# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")

total = 0

if not args['only_qa']:
    # loop the URLs
    for url in rows:
        try:
            # try to download the image
            r = requests.get(url, timeout=60)

            # save the image to disk
            p = os.path.sep.join([args["output"], "{}.jpg".format(
                str(total).zfill(8))])
            #print('Opening the filepath: ' + p)
            with open(p, "wb") as f:
                f.write(r.content)

            # update the counter
            print("[INFO] downloaded: {}".format(p))
            total += 1

        # handle if any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading {}...skipping".format(p))

# loop over the image paths we just downloaded
for imagePath in paths.list_images(args["output"]):
    # initialize if the image should be deleted or not
    delete = False

    # try to load the image
    try:
        image = cv2.imread(imagePath)

        # if the image is `None` then we could not properly load it
        # from disk, so delete it
        if image is None:
            delete = True
            print ('Image deleted because OpenCV could not open it.')
        elif args["qa"] is True:
            cv2.imshow('downloaded image', image)
            key = cv2.waitKey(0) & 255 # have to mask the other bits to get ascii
            if key is not ord('k'): delete = True
            cv2.destroyAllWindows()

    # if OpenCV cannot load the image then the image is likely
    # corrupt so we should delete it
    except:
        print("Except")
        delete = True

    # check to see if the image should be deleted
    if delete:
        print("[INFO] deleting {}".format(imagePath))
        os.remove(imagePath)