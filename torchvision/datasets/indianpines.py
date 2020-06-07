from .utils import loadmat,select_small_cubic,sampling
from .vision import VisionDataset
import os
class IndianPines(VisionDataset):
    """`Indian Pines  <http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Indian_Pines>` Dataset.
    Args:
        root (string): Root directory of dataset where directory
              `IP` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        split (float,optional): Mentions the Validation Split. If split = 0.95 ,Training is 5% of the Total.
        
        PATCH_LENGTH (int,optional): Mentions the Window patch given by 2*PATCH_LENGTH + 1
    """
    base_folder = 'IP'
    
    ipath = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
    ifile = "Indian_pines_corrected.mat"
    imd5 = '66dbc9f4a9b7c9b1445f60a87b505101'
    
    lpath = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_gt.mat"
    lfile = "Indian_pines_gt.mat"
    lmd5 = '9414943dac1d80faaa9165c8b460510c'
    
    def __init__(self, root, train=True,
                 download=False,split=0.95,PATCH_LENGTH=5):

        super(IndianPines, self).__init__(root)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        self.split = split
        self.PATCH_LENGTH = PATCH_LENGTH
        self.TOTAL_SIZE = 10249
        self.data_hsi, self.gt_hsi, self.TOTAL_SIZE, self.TRAIN_SIZE,self.VALIDATION_SPLIT = loadmat(self.ipath,self.lpath,self.imd5,self.lmd5,self.TOTAL_SIZE,self.split)
        self.data = self.data_hsi.reshape(np.prod(self.data_hsi.shape[:2]), np.prod(self.data_hsi.shape[2:]))
        self.gt = self.gt_hsi.reshape(np.prod(self.gt_hsi.shape[:2]),)
        self.CLASSES_NUM = max(self.gt)
        self.INPUT_DIMENSION = self.data_hsi.shape[2]
        self.ALL_SIZE = self.data_hsi.shape[0] * self.data_hsi.shape[1]
        self.VAL_SIZE = int(self.TRAIN_SIZE)
        self.TEST_SIZE = self.TOTAL_SIZE - self.TRAIN_SIZE
        self.data_ = self.data.reshape(self.data_hsi.shape[0], self.data_hsi.shape[1], self.data_hsi.shape[2])
        self.whole_data = self.data_
        self.padded_data = np.lib.pad(self.whole_data, ((self.PATCH_LENGTH, self.PATCH_LENGTH), (      self.PATCH_LENGTH, self.PATCH_LENGTH), (0,0)),'constant', constant_values=0)

        if self.train:
          self.train_indices, _ = sampling(self.VALIDATION_SPLIT, self.gt)
          _, self.total_indices = sampling(1, self.gt)
          self.TRAIN_SIZE = len(self.train_indices)
          self.y_train = self.gt[self.train_indices] - 1
          self.train_data = select_small_cubic(self.TRAIN_SIZE, self.train_indices, self.whole_data,self.PATCH_LENGTH,self.padded_data,self.INPUT_DIMENSION)
          self.x_train = self.train_data.reshape(self.train_data.shape[0], self.train_data.shape[1], self.train_data.shape[2], self.INPUT_DIMENSION)
          self.x__tensor = np.squeeze(self.x_train,axis=1)
          self.y__tensor = np.squeeze(self.y_train,axis=1)
          
          #do stuff only for train
        else:
          self.train_indices, self.test_indices = sampling(self.VALIDATION_SPLIT, self.gt)
          _, self.total_indices = sampling(1, self.gt)
          self.TRAIN_SIZE = len(self.train_indices) 
          self.TEST_SIZE = self.TOTAL_SIZE - self.TRAIN_SIZE
          self.VAL_SIZE = int(self.TRAIN_SIZE)
          self.y_test = self.gt[self.test_indices] - 1
          self.test_data = select_small_cubic(self.TEST_SIZE, self.test_indices, self.whole_data,
                                                       self.PATCH_LENGTH, self.padded_data, self.INPUT_DIMENSION)
          self.x_test_all = self.test_data.reshape(self.test_data.shape[0], self.test_data.shape[1], self.test_data.shape[2], self.INPUT_DIMENSION)
          self.x_test = self.x_test_all[:-self.VAL_SIZE]
          self.y_test = self.y_test[:-self.VAL_SIZE]
          
          self.x__tensor = np.squeeze(self.x_test,axis=1)
          self.y__tensor = np.squeeze(self.y_test,axis=1)
    
          #do stuff only for test
              
        #common code
        
    def __getitem__(self, index):
      return self.x__tensor[index,...],self.y__tensor[index,...]

    def __len__(self):
      return self.x__tensor.shape[0]

    def _check_integrity(self):
        root = self.root
        ifilename, imd5 = self.ifile, self.imd5
        ifpath = os.path.join(root, self.base_folder, ifilename)  
        lfilename, lmd5 = self.lfile, self.lmd5
        lfpath = os.path.join(root, self.base_folder, lfilename)
        if check_integrity(lfpath, lmd5) and check_integrity(ifpath, imd5):
          return True
        return False

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_url(self.ipath, self.root, filename=self.ifile, md5=self.imd5)
        download_url(self.lpath, self.root, filename=self.lfile, md5=self.lmd5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
