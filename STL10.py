import sys, os, urllib.request, tarfile
import numpy as np

class STL10:
    def __init__(self, download_dir):
        self.binary_dir = os.path.join(download_dir, "stl10_binary")

        if not os.path.exists(download_dir):
            os.mkdir(download_dir)
        if not os.path.exists(self.binary_dir):
            os.mkdir(self.binary_dir)

        # download file
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (source_path,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        source_path = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
        dest_path = os.path.join(download_dir, "stl10_binary.tar.gz")
        if not os.path.exists(dest_path):
            urllib.request.urlretrieve(source_path, filename=dest_path, reporthook=_progress)
            # untar
            with tarfile.open(dest_path, "r:gz") as tar:
                tar.extractall(path=download_dir)

    def get_files(self, target):
        assert target in ["train", "test", "unlabeled"]
        if target in ["train", "test"]:
            images = self.load_images(os.path.join(self.binary_dir, target+"_X.bin"))
            labels = self.load_labels(os.path.join(self.binary_dir, target+"_y.bin"))
        else:
            images = self.load_images(os.path.join(self.binary_dir, target+"_X.bin"))
            labels = None
        return images, labels

    def load_images(self, image_binary):
        with open(image_binary, "rb") as fp:
            images = np.fromfile(fp, dtype=np.uint8)
            images = images.reshape(-1, 3, 96, 96)
            return np.transpose(images, (0, 3, 2, 1))

    def load_labels(self, label_binary):
        with open(label_binary) as fp:
            labels = np.fromfile(fp, dtype=np.uint8)
            return labels.reshape(-1, 1) - 1 # 1-10 -> 0-9
